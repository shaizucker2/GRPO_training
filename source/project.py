"""Single-GPU GSM8K experiment: prepare, train, and evaluate."""
import argparse
import json
import os
from pathlib import Path
import time

from rewards import SYSTEM_PROMPT, accuracy_reward, correct, extract_answer


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    destination = Path(args.data)
    if destination.exists():
        raise SystemExit(f'{destination} already exists; choose a new --data directory.')
    api = HfApi()
    model_revision = api.model_info(args.model).sha
    dataset_revision = api.dataset_info('openai/gsm8k').sha
    raw = load_dataset('openai/gsm8k', 'main', revision=dataset_revision)

    def convert(row, index):
        answer = extract_answer(row['answer'])
        if answer is None:
            raise ValueError(f'Cannot parse reference answer at row {index}')
        return {
            'id': index,
            'prompt': [{'role': 'system', 'content': SYSTEM_PROMPT},
                       {'role': 'user', 'content': row['question']}],
            'answer': str(answer),
        }

    train = raw['train'].map(convert, with_indices=True, remove_columns=raw['train'].column_names)
    # Reserve validation from training; official test stays untouched.
    split = train.train_test_split(test_size=256, seed=42)
    test = raw['test'].map(convert, with_indices=True, remove_columns=raw['test'].column_names)
    destination.mkdir(parents=True)
    for name, dataset in [('train', split['train']), ('validation', split['test']), ('test', test)]:
        dataset.save_to_disk(str(destination / name))
    manifest = {
        'model_id': args.model, 'model_revision': model_revision,
        'dataset_id': 'openai/gsm8k', 'dataset_revision': dataset_revision,
        'split_seed': 42, 'system_prompt': SYSTEM_PROMPT,
        'counts': {'train': len(split['train']), 'validation': len(split['test']), 'test': len(test)},
    }
    write_json(destination / 'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


def load_assets(args):
    from datasets import load_from_disk
    manifest = json.loads((Path(args.data) / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['system_prompt'] != SYSTEM_PROMPT:
        raise SystemExit('Prompt changed since prepare; restore it or prepare a new experiment.')
    return manifest, load_from_disk


def load_model(manifest):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Check nvidia-smi and your PyTorch installation.')
    if not torch.cuda.is_bf16_supported():
        raise SystemExit('This configuration requires BF16 support (available on A40).')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise SystemExit('Run with python on one GPU; this starter is configured for one GPU.')
    tokenizer = AutoTokenizer.from_pretrained(manifest['model_id'], revision=manifest['model_revision'])
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        manifest['model_id'], revision=manifest['model_revision'],
        torch_dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to('cuda')
    return model, tokenizer


def train(args):
    import torch
    from peft import LoraConfig
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Output is nonempty; choose a fresh --output directory.')
    if args.steps < 1 or args.hours <= 0:
        raise SystemExit('--steps and --hours must be positive.')
    set_seed(42)
    manifest, load_dataset = load_assets(args)
    dataset = load_dataset(str(Path(args.data) / 'train'))
    model, tokenizer = load_model(manifest)
    # Refuse silently truncated training prompts.
    longest = max(len(tokenizer.apply_chat_template(p, add_generation_prompt=True)) for p in dataset['prompt'])
    if longest > 1024:
        raise SystemExit(f'Longest prompt is {longest} tokens; increase max_prompt_length in the script.')

    class TimeBudget(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.monotonic()

        def on_step_end(self, training_args, state, control, **kwargs):
            if time.monotonic() - self.started >= args.hours * 3600:
                control.should_training_stop = True
                control.should_save = True
            return control

    config = GRPOConfig(
        output_dir=str(output), bf16=True,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_generations=4, steps_per_generation=8,
        max_prompt_length=1024, max_completion_length=args.tokens,
        max_steps=args.steps, learning_rate=5e-6, warmup_ratio=0.05,
        lr_scheduler_type='constant_with_warmup',
        beta=0.04, temperature=0.8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        use_vllm=False, loss_type='grpo',
        mask_truncated_completions=True,
        logging_steps=1, save_steps=50, save_total_limit=2,
        report_to='none', seed=42, data_seed=42,
    )
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, args=config,
        train_dataset=dataset, reward_funcs=accuracy_reward,
        peft_config=LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            task_type='CAUSAL_LM',
        ),
        callbacks=[TimeBudget()],
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'run.json', {'assets': manifest, 'arguments': vars(args)})
    started = time.monotonic()
    result = trainer.train()
    final = output / 'final'
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    write_json(final / 'assets.json', manifest)
    metrics = {**result.metrics, 'elapsed_seconds': time.monotonic() - started,
               'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
    trainer.save_state()
    write_json(output / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2))


def evaluate(args):
    import torch
    from transformers import set_seed
    set_seed(42)
    destination = Path(args.output)
    if destination.exists() or destination.with_suffix('.jsonl').exists():
        raise SystemExit('Evaluation output already exists; choose another --output filename.')
    manifest, load_dataset = load_assets(args)
    dataset = load_dataset(str(Path(args.data) / args.split))
    if args.limit < 0:
        raise SystemExit('--limit must be nonnegative (0 means the whole split).')
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if args.adapter:
        saved = json.loads((Path(args.adapter) / 'assets.json').read_text(encoding='utf-8'))
        if saved != manifest:
            raise SystemExit('Adapter and evaluation data/model manifests differ.')
    model, tokenizer = load_model(manifest)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.config.use_cache = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    totals = {'correct': 0, 'formatted': 0, 'truncated': 0}
    started = time.monotonic()
    with destination.with_suffix('.jsonl').open('x', encoding='utf-8') as log:
        for index, example in enumerate(dataset):
            prompt = tokenizer.apply_chat_template(example['prompt'], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
            with torch.inference_mode():
                result = model.generate(**inputs, do_sample=False, max_new_tokens=args.tokens,
                                        pad_token_id=tokenizer.pad_token_id)
            generated = result[0, inputs['input_ids'].shape[1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            eos = model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            truncated = len(generated) >= args.tokens and int(generated[-1]) not in eos_ids
            score = correct(text, example['answer'])
            totals['correct'] += int(score)
            totals['formatted'] += int(extract_answer(text) is not None)
            totals['truncated'] += int(truncated)
            log.write(json.dumps({'id': example['id'], 'question': example['prompt'][-1]['content'],
                                  'gold': example['answer'], 'completion': text,
                                  'correct': score, 'truncated': truncated}) + '\n')
            log.flush()
            if (index + 1) % 25 == 0:
                print(f'{index + 1}/{len(dataset)} correct={totals["correct"]}', flush=True)
    metrics = {'n': len(dataset), **totals, 'accuracy': totals['correct'] / len(dataset),
               'format_rate': totals['formatted'] / len(dataset),
               'truncation_rate': totals['truncated'] / len(dataset),
               'elapsed_seconds': time.monotonic() - started,
               'assets': manifest, 'arguments': vars(args)}
    write_json(destination, metrics)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--data', default='data')
    t = sub.add_parser('train')
    t.add_argument('--data', default='data')
    t.add_argument('--output', default='runs/grpo')
    t.add_argument('--steps', type=int, default=1000)
    t.add_argument('--hours', type=float, default=3.0)
    t.add_argument('--tokens', type=int, default=512)
    e = sub.add_parser('evaluate')
    e.add_argument('--data', default='data')
    e.add_argument('--adapter')
    e.add_argument('--split', choices=['validation', 'test'], default='validation')
    e.add_argument('--limit', type=int, default=0)
    e.add_argument('--tokens', type=int, default=512)
    e.add_argument('--output', required=True)
    args = parser.parse_args()
    if hasattr(args, 'tokens') and args.tokens < 1:
        parser.error('--tokens must be positive')
    {'prepare': prepare, 'train': train, 'evaluate': evaluate}[args.command](args)


if __name__ == '__main__':
    main()
