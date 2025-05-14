print(r"""
  _____    ____   __      __  _            _____   __     __   _____   _______   ______   __  __ 
 / ____|  / __ \  \ \    / / | |          / ____|  \ \   / /  / ____| |__   __| |  ____| |  \/  |
| (___   | |  | |  \ \  / /  | |         | (___     \ \_/ /  | (___      | |    | |__    | \  / |
 \___ \  | |  | |   \ \/ /   | |          \___ \     \   /    \___ \     | |    |  __|   | |\/| |
 ____) | | |__| |    \  /    | |____      ____) |     | |     ____) |    | |    | |____  | |  | |
|_____/   \____/      \/     |______|    |_____/      |_|    |_____/     |_|    |______| |_|  |_|
""")
print("https://github.com/oppolla/Self-Organizing-Virtual-Lifeform")
print("Version 0.1.1\n")
print("License: MIT\n")

import sys
import platform
import os

MIN_PYTHON = (3, 8)
if sys.version_info < MIN_PYTHON:
    print(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required. You have {platform.python_version()}.")
    response = input("Type 'c' to continue at your own risk, or 'n' to exit. [c/n]: ").strip().lower()
    if response == 'c':
        print("Continuing at your own risk...")
    else:
        sys.exit(1)

import subprocess

REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "pydantic",
    "numpy"
]

def check_dependencies():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("You can install them with:")
        print(f"    pip install {' '.join(missing)}\n")
        response = input("Type 'i' to install automatically, 'm' to manually install and exit, 'c' to continue at your own risk, or 'n' to exit. [i/m/c/n]: ").strip().lower()
        if response == "i":
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
                print("All required packages installed.")
                rerun = input("Would you like to re-run the entry point now? (y/n): ").strip().lower()
                if rerun == "y":
                    print("Re-running the entry point...\n")
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                else:
                    print("Please re-run the script manually to continue.")
            except Exception as e:
                print(f"[ERROR] Automatic installation failed: {e}")
            sys.exit(1)
        elif response == "m":
            print("Please type the above pip install command in your terminal, then re-run this script.")
            sys.exit(1)
        elif response == "c":
            print("Continuing at your own risk...")
            return
        else:
            print("Exiting. Please install the missing packages and try again.")
            sys.exit(1)

check_dependencies()

import argparse
import traceback
from sovl_conductor import SOVLOrchestrator
from sovl_cli import run_cli

class SOVLInitializer:
    """
    Minimal, canonical system initializer for SOVL.
    Responsible for parsing CLI args, initializing the system, and launching the CLI.
    """
    def __init__(self):
        self.system = None
        self.context = None

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Run the SOVL AI system",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("--config", default="sovl_config.json", help="Path to configuration file")
        parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use for computation")
        parser.add_argument("--log-file", default=None, help="Path to log file (optional)")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
        return parser.parse_args()

    def initialize_system(self, args):
        # Pass log_file if provided, else let orchestrator use its default
        log_file = args.log_file if args.log_file else None
        try:
            if log_file:
                orchestrator = SOVLOrchestrator(config_path=args.config, log_file=log_file)
            else:
                orchestrator = SOVLOrchestrator(config_path=args.config)
            orchestrator.initialize_system()
            # Canonical: orchestrator._system is the initialized SOVLSystem
            self.system = getattr(orchestrator, '_system', None)
            self.context = getattr(self.system, 'context', None) if self.system else None
            return True
        except Exception as e:
            print(f"[ERROR] System initialization failed: {e}")
            print(traceback.format_exc())
            return False

    def run(self):
        args = self.parse_args()
        if self.initialize_system(args):
            run_cli(self.system)
        else:
            print("[ERROR] System initialization failed. CLI will not be started.")

if __name__ == "__main__":
    print("Commencing incarnation process...")
    SOVLInitializer().run()
