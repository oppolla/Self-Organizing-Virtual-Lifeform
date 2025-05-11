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
    SOVLInitializer().run()
