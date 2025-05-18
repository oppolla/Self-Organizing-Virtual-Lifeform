import os
import sys
import time
import importlib.util
import traceback
from typing import Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json
from glob import glob

from sovl_config import ConfigManager
from sovl_logger import Logger
from sovl_error import ErrorManager

@dataclass
class TestResult:
    name: str
    status: str  # 'PASS' or 'FAIL'
    duration: float
    error: Optional[str] = None
    details: Optional[Dict] = None
    category: Optional[str] = None
    test_class: Optional[str] = None

class SOVLTestRunner:
    """Test runner for SOVL system tests."""
    
    def __init__(self, sovl_system, verbose: bool = False, config: Dict = None):
        self.sovl_system = sovl_system
        self.verbose = verbose
        
        # Get the system's config manager
        self.config_manager = getattr(sovl_system, 'config_handler', None)
        if not self.config_manager:
            # If no config manager in system, create a new one
            self.config_manager = ConfigManager()
        
        # Initialize logger
        self.logger = Logger.instance()
        
        # Get test-specific configuration
        test_config = self.config_manager.get('test_config', {})
        if not test_config:
            # Set default test configuration if not present
            test_config = {
                'test_dir': 'tests',
                'output_dir': 'test_results',
                'parallel': False,
                'no_color': False,
                'test_patterns': ['test_*.py', '*_test.py', '*_stress.py'],
                'slow_test_threshold': 1.0,
                'save_results': True,
                'cleanup_on_exit': True,
                'state_preservation': True,
                'max_retries': 3,
                'retry_delay': 1.0
            }
        
        # Override with provided config if any
        if config:
            test_config.update(config)
        
        # Set instance attributes from config
        self.test_dir = Path(test_config.get('test_dir', 'tests'))
        self.output_dir = Path(test_config.get('output_dir', 'test_results'))
        self.parallel = test_config.get('parallel', False)
        self.no_color = test_config.get('no_color', False)
        self.test_patterns = test_config.get('test_patterns', ['test_*.py', '*_test.py', '*_stress.py'])
        self.slow_test_threshold = test_config.get('slow_test_threshold', 1.0)
        self.save_results = test_config.get('save_results', True)
        self.cleanup_on_exit = test_config.get('cleanup_on_exit', True)
        self.state_preservation = test_config.get('state_preservation', True)
        self.max_retries = test_config.get('max_retries', 3)
        self.retry_delay = test_config.get('retry_delay', 1.0)
        
        # Initialize error manager if available
        self.error_manager = getattr(sovl_system, 'error_manager', None)
        
        self.colors = {
            'green': '\033[92m' if not self.no_color else '',
            'red': '\033[91m' if not self.no_color else '',
            'yellow': '\033[93m' if not self.no_color else '',
            'blue': '\033[94m' if not self.no_color else '',
            'reset': '\033[0m' if not self.no_color else ''
        }

    def _log_error(self, error_type: str, error: Exception, context: str = None):
        """Log errors using the system's error manager if available."""
        if self.error_manager:
            self.error_manager.record_error(
                error_type=error_type,
                message=str(error),
                stack_trace=traceback.format_exc(),
                context=context or "SOVLTestRunner"
            )
        if self.logger:
            self.logger.log_error(f"Test error ({error_type}): {error}")
        if self.verbose:
            print(f"Error ({error_type}): {error}")
            print(traceback.format_exc())

    def _log_event(self, event_type: str, details: Dict = None):
        """Log events using the system's logger if available."""
        if self.logger:
            self.logger.log_event(
                event_type=event_type,
                details=details or {},
                source="SOVLTestRunner"
            )

    def discover_tests(self) -> Dict[str, List[Dict]]:
        """
        Discover all available tests in the test directory.
        Returns a dictionary of test categories and their tests.
        """
        tests = {}
        
        if not self.test_dir.exists():
            if self.verbose:
                print(f"Warning: Test directory {self.test_dir} does not exist")
            return tests

        for root, dirs, files in os.walk(self.test_dir):
            root_path = Path(root)
            
            # Skip __pycache__ and virtual environments
            if any(skip in root for skip in ['__pycache__', 'venv', '.env']):
                continue
                
            category = root_path.relative_to(self.test_dir).parts[0] if root_path != self.test_dir else "uncategorized"
            
            if category not in tests:
                tests[category] = []

            for pattern in self.test_patterns:
                matches = glob(str(root_path / pattern))
                for match in matches:
                    file_path = Path(match)
                    if file_path.is_file():
                        test_info = self._get_test_info(file_path, category)
                        if test_info:
                            tests[category].append(test_info)

        return tests

    def _get_test_info(self, file_path: Path, category: str) -> Optional[Dict]:
        """Extract test information from a test file."""
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for test classes (classes with run() method or unittest.TestCase subclasses)
            test_classes = []
            for item_name in dir(module):
                item = getattr(module, item_name)
                if isinstance(item, type):
                    if hasattr(item, 'run') or 'TestCase' in [base.__name__ for base in item.__bases__]:
                        test_classes.append(item_name)

            if test_classes:
                return {
                    'name': file_path.stem,
                    'path': str(file_path),
                    'category': category,
                    'description': module.__doc__ or "No description available",
                    'test_classes': test_classes,
                    'has_setup': hasattr(module, 'setup'),
                    'has_teardown': hasattr(module, 'teardown')
                }
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load test file {file_path}: {e}")
        return None

    def _filter_tests(self, tests: Dict[str, List[Dict]], pattern: str = None, test_name: str = None) -> List[Dict]:
        """Filter tests based on pattern or specific test name."""
        tests_to_run = []
        for category, category_tests in tests.items():
            for test in category_tests:
                if (pattern and pattern.lower() in test['name'].lower()) or \
                   (test_name and test_name == test['name']) or \
                   (not pattern and not test_name):
                    tests_to_run.append(test)
        return tests_to_run

    def run_tests(self, pattern: str = None, test_name: str = None) -> Dict:
        """Run tests with progress reporting."""
        start_time = time.time()
        results = []
        tests = self.discover_tests()
        tests_to_run = self._filter_tests(tests, pattern, test_name)
        
        total_tests = len(tests_to_run)
        if total_tests == 0:
            print("No tests found matching criteria")
            return self._create_empty_result()
        
        print(f"\nRunning {total_tests} tests...")
        
        for i, test in enumerate(tests_to_run, 1):
            if self.verbose:
                print(f"\n[{i}/{total_tests}] Running {test['name']}...")
            else:
                self._update_progress(i, total_tests)
                
            result = self._run_single_test(test)
            results.append(result)
        
        print("\n")  # Clear progress line
        
        # Analyze results
        duration = time.time() - start_time
        analysis = self.analyze_results({'results': results, 'duration': duration})
        
        return {
            'results': results,
            'summary': self._create_summary(results, duration),
            'analysis': analysis,
            'formatted_output': self.format_results(results, analysis)
        }

    def _run_single_test(self, test_info: Dict) -> TestResult:
        """Run a single test file and return its result."""
        start_time = time.time()
        original_state = None
        
        try:
            # Save system state if possible
            if hasattr(self.sovl_system, 'get_state'):
                original_state = self.sovl_system.get_state()
            
            # Import test module
            spec = importlib.util.spec_from_file_location(test_info['name'], test_info['path'])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Module level setup
            if test_info['has_setup']:
                module.setup(self.sovl_system)
            
            for class_name in test_info['test_classes']:
                test_class = getattr(module, class_name)
                instance = test_class()
                
                # Instance level setup
                if hasattr(instance, 'setup'):
                    instance.setup(self.sovl_system)
                
                try:
                    if hasattr(instance, 'run'):
                        instance.run()
                    elif 'TestCase' in [base.__name__ for base in test_class.__bases__]:
                        import unittest
                        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
                        result = unittest.TextTestRunner(verbosity=2 if self.verbose else 1).run(suite)
                        if not result.wasSuccessful():
                            raise Exception(f"Test failures occurred in {class_name}")
                finally:
                    # Instance level teardown
                    if hasattr(instance, 'teardown'):
                        instance.teardown()
            
            # Module level teardown
            if test_info['has_teardown']:
                module.teardown()
                
            duration = time.time() - start_time
            return TestResult(
                name=test_info['name'],
                status='PASS',
                duration=duration,
                category=test_info['category'],
                test_class=class_name
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_info['name'],
                status='FAIL',
                duration=duration,
                error=str(e),
                details={'traceback': traceback.format_exc()},
                category=test_info['category'],
                test_class=class_name if 'class_name' in locals() else None
            )
        finally:
            # Restore system state if possible
            if original_state and hasattr(self.sovl_system, 'set_state'):
                try:
                    self.sovl_system.set_state(original_state)
                except Exception as e:
                    print(f"Warning: Failed to restore system state: {e}")

    def _update_progress(self, current: int, total: int):
        """Show progress bar for test execution."""
        width = 40
        progress = current / total
        filled = int(width * progress)
        bar = '=' * filled + '-' * (width - filled)
        print(f'\r[{bar}] {current}/{total} tests completed', end='', flush=True)

    def analyze_results(self, results: Dict) -> Dict:
        """Analyze test results for patterns and insights."""
        analysis = {
            'slow_tests': [],
            'frequent_failures': {},
            'error_patterns': {},
            'recommendations': [],
            'categories': {}
        }
        
        for result in results['results']:
            # Track category statistics
            category = result.category or 'uncategorized'
            if category not in analysis['categories']:
                analysis['categories'][category] = {'total': 0, 'passed': 0, 'failed': 0}
            analysis['categories'][category]['total'] += 1
            analysis['categories'][category]['passed' if result.status == 'PASS' else 'failed'] += 1
            
            # Track slow tests
            if result.duration > self.slow_test_threshold:
                analysis['slow_tests'].append({
                    'name': result.name,
                    'duration': result.duration,
                    'category': category
                })
            
            # Track error patterns
            if result.status == 'FAIL':
                error_type = type(result.error).__name__ if result.error else 'Unknown'
                analysis['error_patterns'][error_type] = analysis['error_patterns'].get(error_type, 0) + 1
        
        # Generate recommendations
        if analysis['slow_tests']:
            analysis['recommendations'].append(
                f"Found {len(analysis['slow_tests'])} slow tests (>{self.slow_test_threshold}s). "
                "Consider optimization or parallel execution."
            )
        
        if analysis['error_patterns']:
            common_errors = sorted(analysis['error_patterns'].items(), key=lambda x: x[1], reverse=True)
            analysis['recommendations'].append(
                f"Most common error type: {common_errors[0][0]} ({common_errors[0][1]} occurrences)"
            )
        
        return analysis

    def format_results(self, results: List[TestResult], analysis: Dict) -> str:
        """Enhanced formatting with error categorization and analysis."""
        output = ["\n=== SOVL Test Results ===\n"]
        
        # Group results by category and status
        by_category = {}
        for result in results:
            category = result.category or 'uncategorized'
            if category not in by_category:
                by_category[category] = {'passed': [], 'failed': []}
            by_category[category]['passed' if result.status == 'PASS' else 'failed'].append(result)
        
        # Show results by category
        for category, cat_results in by_category.items():
            output.append(f"\n{category.upper()}:")
            
            if cat_results['failed']:
                output.append("\nFailed Tests:")
                for result in cat_results['failed']:
                    output.extend(self._format_failed_test(result))
            
            if cat_results['passed']:
                output.append("\nPassed Tests:")
                for result in cat_results['passed']:
                    output.extend(self._format_passed_test(result))
        
        # Add analysis section
        if analysis['slow_tests'] or analysis['error_patterns']:
            output.append("\nAnalysis:")
            
            if analysis['slow_tests']:
                output.append("\nSlow Tests:")
                for test in analysis['slow_tests']:
                    output.append(f"  - {test['name']} ({test['duration']:.2f}s)")
            
            if analysis['error_patterns']:
                output.append("\nError Patterns:")
                for error_type, count in analysis['error_patterns'].items():
                    output.append(f"  - {error_type}: {count} occurrences")
            
            if analysis['recommendations']:
                output.append("\nRecommendations:")
                for rec in analysis['recommendations']:
                    output.append(f"  - {rec}")
        
        return "\n".join(output)

    def _format_failed_test(self, result: TestResult) -> List[str]:
        """Format a failed test result."""
        output = [
            f"  {self.colors['red']}{result.name}: FAIL{self.colors['reset']}",
            f"    Duration: {result.duration:.2f}s",
            f"    Error: {result.error}"
        ]
        if self.verbose and result.details:
            output.extend([
                "    Traceback:",
                result.details['traceback']
            ])
        return output

    def _format_passed_test(self, result: TestResult) -> List[str]:
        """Format a passed test result."""
        return [
            f"  {self.colors['green']}{result.name}: PASS{self.colors['reset']}",
            f"    Duration: {result.duration:.2f}s"
        ]

    def _create_summary(self, results: List[TestResult], duration: float) -> Dict:
        """Create a summary of test results."""
        return {
            'total': len(results),
            'passed': sum(1 for r in results if r.status == 'PASS'),
            'failed': sum(1 for r in results if r.status == 'FAIL'),
            'duration': duration
        }

    def _create_empty_result(self) -> Dict:
        """Create an empty result set."""
        return {
            'results': [],
            'summary': {'total': 0, 'passed': 0, 'failed': 0, 'duration': 0},
            'analysis': {'slow_tests': [], 'error_patterns': {}, 'recommendations': []},
            'formatted_output': "No tests were run."
        }

    def get_test_help(self) -> str:
        """Return help text for test command."""
        return """
Test Command Usage:
-----------------
/test                 Show this help message
/test run             Run all tests
/test list            List available tests
/test <test_name>     Run specific test
/test -v              Run all tests with verbose output
/test -p <pattern>    Run tests matching pattern

Examples:
  /test run                       # Run all tests
  /test curiosity_test            # Run specific test
  /test -p curiosity              # Run all tests with 'curiosity' in name
  /test -v                        # Run all tests with verbose output
"""

    def get_completions(self, text: str) -> List[str]:
        """Return possible completions for the given text."""
        commands = ['all', 'list', '-v', '-p']
        
        # Add all test names
        for category_tests in self.discover_tests().values():
            commands.extend(test['name'] for test in category_tests)
            
        # Filter based on current text
        return [cmd for cmd in commands if cmd.startswith(text)]

    def save_results(self, results: Dict, output_dir: str = None):
        """Save test results to a file."""
        output_dir = Path(output_dir or self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
        
        # Convert TestResult objects to dictionaries
        serializable_results = []
        for result in results['results']:
            result_dict = {
                'name': result.name,
                'status': result.status,
                'duration': result.duration,
                'error': result.error,
                'details': result.details,
                'category': result.category,
                'test_class': result.test_class
            }
            serializable_results.append(result_dict)
            
        output = {
            'results': serializable_results,
            'summary': results['summary'],
            'analysis': results['analysis'],
            'timestamp': timestamp,
            'config': self.config_manager.get('test_config', {})
        }
        
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
            
        if self.verbose:
            print(f"\nTest results saved to: {filepath}")

    def cleanup(self):
        """Cleanup resources after testing."""
        if self.cleanup_on_exit:
            try:
                # Cleanup temporary files
                if self.output_dir.exists():
                    for temp_file in self.output_dir.glob("*_temp_*"):
                        temp_file.unlink()
                
                self._log_event("test_cleanup_completed")
            except Exception as e:
                self._log_error("cleanup_error", e)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        if exc_type:
            self._log_error("test_runner_error", exc_val)
