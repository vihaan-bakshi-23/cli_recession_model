# File: src/10_monthly_refresh.py
# Run from: cli_recession_model/ root
# Usage: python src/10_monthly_refresh.py            (runs core refresh: 1-6, 8)
#        python src/10_monthly_refresh.py --full      (also re-runs 7 and 7b)

import subprocess
import sys
import datetime
import argparse
import os
import time

CORE_PIPELINE = [
    "src/1_data_pull.py",
    "src/2_preprocessing.py",
    "src/3_hp_filter.py",
    "src/4_normalize_leads.py",
    "src/5_build_cli.py",
    "src/6_recession_probability.py",
    "src/8_dashboard.py",
]

FULL_EXTRAS = [
    "src/7_backtesting.py",
    "src/7b_walkforward.py",
]

LOG_FILE = "logs/refresh_log.txt"

def run_step(script_path, log_f):
    print(f"Running {script_path} ...")
    start = time.time()

    log_f.write(f"\n--- Running {script_path} at {datetime.datetime.now()} ---\n")
    log_f.flush()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, encoding="utf-8",
        env=env
    )
    log_f.write(result.stdout)
    log_f.write(result.stderr)
    log_f.flush()

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"FAILED at {script_path} after {elapsed:.1f}s. See {LOG_FILE} for details.")
        sys.exit(1)
    else:
        print(f"OK: {script_path} ({elapsed:.1f}s)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Also re-run backtesting and walk-forward validation")
    args = parser.parse_args()

    import os
    os.makedirs("logs", exist_ok=True)

    steps = CORE_PIPELINE + FULL_EXTRAS if args.full else CORE_PIPELINE

    with open(LOG_FILE, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n=== Refresh started {datetime.datetime.now()} ===\n")
        for step in steps:
            run_step(step, log_f)
        log_f.write(f"=== Refresh completed successfully {datetime.datetime.now()} ===\n")

    print("\nRefresh complete. Dashboard updated at outputs/recession_dashboard.html")

if __name__ == "__main__":
    main()