"""Linux launcher: python3 launch.py start | status | logs | results."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
HOME = ROOT / 'experiments'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def latest():
    pointer = HOME / 'latest.txt'
    if not pointer.exists():
        raise SystemExit('No experiment yet. Run: python3 launch.py start')
    return HOME / pointer.read_text().strip()


def acquire_lock():
    import fcntl
    HOME.mkdir(exist_ok=True)
    handle = (HOME / 'active.lock').open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit('An experiment is already running. Use: python3 launch.py status')
    return handle


def start():
    if sys.platform != 'linux':
        raise SystemExit('Run this launcher on your Linux GPU server.')
    if sys.version_info[:2] not in [(3, 11), (3, 12)]:
        raise SystemExit('Use Python 3.11 or 3.12, e.g. python3.11 launch.py start')
    lock = acquire_lock()
    run = HOME / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    run.mkdir()
    with (run / 'run.log').open('w') as log:
        process = subprocess.Popen(
            [sys.executable, '-u', str(Path(__file__).resolve()), '_worker',
             '--run', str(run), '--lock-fd', str(lock.fileno())],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, pass_fds=(lock.fileno(),),
        )
    # The worker inherits the locked descriptor; closing our copy keeps its lock.
    lock.close()
    (HOME / 'latest.txt').write_text(run.name, encoding='utf-8')
    print(f'Started PID {process.pid}. Requested physical GPU index: 2.')
    print(f'Log: {run / "run.log"}')
    print('Check progress: python3 launch.py status')
    print('Watch log:      python3 launch.py logs')
    print('You can disconnect SSH; the job runs in the background.')


def summary(run):
    lines = ['GSM8K before/after results', '']
    for split in ['validation', 'test']:
        before = json.loads((run / f'baseline-{split}.json').read_text())
        after = json.loads((run / f'trained-{split}.json').read_text())
        if before['assets'] != after['assets'] or before['n'] != after['n']:
            raise ValueError('Evaluation datasets do not match')
        delta = 100 * (after['accuracy'] - before['accuracy'])
        lines.extend([
            f'{split}: {before["n"]} questions',
            f'  Before: {100 * before["accuracy"]:.2f}%',
            f'  After:  {100 * after["accuracy"]:.2f}%',
            f'  Change: {delta:+.2f} percentage points',
            f'  Answer format: {before["format_rate"]:.1%} -> {after["format_rate"]:.1%}',
            f'  Truncation: {before["truncation_rate"]:.1%} -> {after["truncation_rate"]:.1%}', '',
        ])
    lines.append('Training improvement is not guaranteed; inspect saved answers as well as scores.')
    text = '\n'.join(lines) + '\n'
    (run / 'summary.txt').write_text(text, encoding='utf-8')
    print(text, flush=True)


def worker(args):
    # Keep the inherited file descriptor open for the entire experiment.
    os.fstat(args.lock_fd)
    run = Path(args.run)
    state = {'pid': os.getpid(), 'state': 'running', 'stage': 'starting', 'gpu_index': 2}
    env = dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')

    def stage(label, command):
        state.update(stage=label, updated=datetime.now().isoformat())
        save(run / 'status.json', state)
        print(f'\n=== {label} ===', flush=True)
        subprocess.run([str(x) for x in command], cwd=ROOT, env=env, check=True)

    try:
        save(run / 'status.json', state)
        # Resolve nvidia-smi's physical index to a UUID to avoid CUDA index-order ambiguity.
        uuid = subprocess.check_output(
            ['nvidia-smi', '-i', '2', '--query-gpu=uuid', '--format=csv,noheader'], text=True
        ).strip()
        if not uuid.startswith('GPU-') or '\n' in uuid:
            raise RuntimeError(f'Could not identify physical GPU 2: {uuid!r}')
        env['CUDA_VISIBLE_DEVICES'] = uuid
        state['gpu_uuid'] = uuid
        print(f'Using ONLY physical GPU 2: {uuid}', flush=True)
        stage('GPU details', ['nvidia-smi', '-i', '2'])
        python = ROOT / '.venv' / 'bin' / 'python'
        if not python.exists():
            stage('Create virtual environment', [sys.executable, '-m', 'venv', ROOT / '.venv'])
        stage('Install pip', [python, '-m', 'pip', 'install', '--upgrade', 'pip'])
        stage('Install CUDA PyTorch', [python, '-m', 'pip', 'install', 'torch==2.8.0',
                                      '--index-url', 'https://download.pytorch.org/whl/cu126'])
        stage('Install training dependencies', [python, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        stage('Check dependencies', [python, '-m', 'pip', 'check'])
        stage('Check GPU in PyTorch', [python, '-c',
            'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; '
            'assert torch.cuda.device_count() == 1, "Expected one visible GPU"; '
            'assert torch.cuda.is_bf16_supported(), "BF16 unavailable"; '
            'x=torch.ones(1, device="cuda"); print("GPU:", torch.cuda.get_device_name(0)); '
            'print("Free/total bytes:", torch.cuda.mem_get_info())'])
        stage('Reward tests', [python, '-m', 'unittest', '-v', 'test_rewards.py'])
        with (run / 'installed-packages.txt').open('w') as output:
            subprocess.run([str(python), '-m', 'pip', 'freeze'], stdout=output, env=env, check=True)
        data = run / 'data'
        base = [python, '-u', 'project.py']
        stage('Prepare GSM8K', base + ['prepare', '--data', data])
        stage('Two-step training smoke test', base + ['train', '--data', data, '--steps', '2',
              '--hours', '0.25', '--output', run / 'smoke'])
        stage('Check saved smoke adapter', base + ['evaluate', '--data', data, '--limit', '4',
              '--adapter', run / 'smoke' / 'final', '--output', run / 'smoke-evaluation.json'])
        for split in ['validation', 'test']:
            stage(f'Baseline {split} evaluation', base + ['evaluate', '--data', data,
                  '--split', split, '--output', run / f'baseline-{split}.json'])
        stage('GRPO training (up to 3 hours or 1000 steps)', base + ['train', '--data', data,
              '--hours', '3', '--steps', '1000', '--output', run / 'training'])
        for split in ['validation', 'test']:
            stage(f'Trained {split} evaluation', base + ['evaluate', '--data', data,
                  '--split', split, '--adapter', run / 'training' / 'final',
                  '--output', run / f'trained-{split}.json'])
        summary(run)
        state.update(state='completed', stage='Finished; summary.txt is ready')
    except Exception as error:
        state.update(state='failed', error=str(error))
        print(f'\nFAILED during {state["stage"]}: {error}', flush=True)
        print('Fix the reported error before starting a fresh experiment. Existing results are preserved.', flush=True)
    finally:
        state['updated'] = datetime.now().isoformat()
        save(run / 'status.json', state)
    return 0 if state['state'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['start', 'status', 'logs', 'results', '_worker'])
    parser.add_argument('--run')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.command == 'start':
        start()
    elif args.command == '_worker':
        return worker(args)
    else:
        run = latest()
        if args.command == 'status':
            path = run / 'status.json'
            print(path.read_text() if path.exists() else 'Starting; check again shortly.')
            print(f'Log: {run / "run.log"}')
            if path.exists() and json.loads(path.read_text())['state'] == 'running':
                import fcntl
                with (HOME / 'active.lock').open('a+') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        print('WARNING: worker is no longer running; inspect the log for interruption.')
                    except BlockingIOError:
                        pass
        elif args.command == 'results':
            path = run / 'summary.txt'
            print(path.read_text() if path.exists() else 'Results are not ready. Run: python3 launch.py status')
        elif args.command == 'logs':
            try:
                subprocess.run(['tail', '-n', '60', '-f', str(run / 'run.log')])
            except KeyboardInterrupt:
                print('\nStopped watching. Training continues in the background.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
