"""
Ch04: Dynamic Programming Visualizations  (consolidated wrapper)
Runs all three original DP visualization scripts with output
redirected to asserts/ch04_dp/.

Figures generated (in asserts/ch04_dp/):
  bellman_backup_static.png / bellman_backup_anim.gif
  policy_iteration_static.png / policy_iteration_anim.gif
  value_iteration_static.png / value_iteration_anim.gif
  value_iteration_qvals.png / vi_pi_comparison.png
"""
import os
import sys
import types
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, '..', '..')
CODE_ROOT = os.path.join(REPO_ROOT, 'code')
OUT_DIR = os.path.join(REPO_ROOT, 'asserts', 'ch04_dp')
os.makedirs(OUT_DIR, exist_ok=True)


def _run_script(script_path: str):
    """Load a script as a module, overriding its OUT_DIR before execution."""
    spec = importlib.util.spec_from_file_location('_tmp_mod', script_path)
    mod = types.ModuleType(spec.name)
    # Inject the patched OUT_DIR so the script's module-level code sees it
    mod.__file__ = script_path
    # We need to exec the source with OUT_DIR already set
    with open(script_path) as f:
        source = f.read()
    # Replace the OUT_DIR assignment
    source = source.replace(
        'OUT_DIR = os.path.join(SCRIPT_DIR, "..", "asserts")',
        f'OUT_DIR = r"{OUT_DIR}"',
    )
    exec(compile(source, script_path, 'exec'), mod.__dict__)


if __name__ == '__main__':
    scripts = [
        os.path.join(CODE_ROOT, 'bellman_backup_viz.py'),
        os.path.join(CODE_ROOT, 'policy_iteration_viz.py'),
        os.path.join(CODE_ROOT, 'value_iteration_viz.py'),
    ]
    for s in scripts:
        print(f"\n--- Running {os.path.basename(s)} ---")
        _run_script(s)
    print("\nCh04 done. Assets in asserts/ch04_dp/")
