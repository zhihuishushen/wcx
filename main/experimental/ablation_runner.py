import subprocess
import os
import json

def run_ablation(samples_dir, mode, out_dir):
    """
    mode: 'lower-only', 'upper-only', 'gated'
    This runner invokes the training entrypoint with flags. For now it calls run_training.py with env var.
    """
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env['ABLATION_MODE'] = mode
    env['SAMPLES_DIR'] = samples_dir
    log_path = os.path.join(out_dir, f"{mode}.log")
    cmd = ['python', os.path.join(os.path.dirname(__file__), '..', '..', 'run_training.py')]
    print(f"Running {mode}, log -> {log_path}")
    with open(log_path, 'w', encoding='utf-8') as logf:
        subprocess.run(cmd, env=env, stdout=logf, stderr=logf)

def main():
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    samples = os.environ.get('SAMPLES_DIR', base)
    out = os.path.join(os.path.dirname(__file__), '..', '..', 'training', 'ablation')
    for mode in ('lower-only', 'upper-only', 'gated'):
        run_ablation(samples, mode, out)

if __name__ == '__main__':
    main()














