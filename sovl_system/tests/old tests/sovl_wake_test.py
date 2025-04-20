from sovl_system.sovl_main import SOVLSystem

"""
This script performs a set of stress tests on the SOVLSystem to evaluate its performance, stability, and error handling.
The tests include:

1. Cold Start Test:
   - Verifies the system's ability to generate output immediately after initialization.
   - Checks for any issues during the initial setup phase.

2. Training Stress Test:
   - Runs a short training cycle using predefined training and validation data.
   - Tests the system's ability to handle training processes without failures.

3. Memory Test:
   - Stresses the memory by repeatedly generating output with large input and large token limits.
   - Evaluates the system's capability to manage memory-intensive tasks.

4. Error Recovery Test:
   - Intentionally provides extreme inputs (e.g., empty input or large token generation requests).
   - Observes the system's ability to handle errors gracefully and recover.

These tests aim to ensure the robustness and reliability of the SOVLSystem under various conditions.
"""

def stress_test():
    # Initialize the SOVL system
    sovl = SOVLSystem()
    
    print("-- Cold Start Test --")
    try:
        # Test generation immediately after initialization
        result = sovl.generate("Hello world", max_new_tokens=20)
        print("Generated:", result)
    except Exception as e:
        print("Error during Cold Start Test:", e)
    
    print("\n-- Training Stress Test --")
    try:
        # Run a short training cycle
        sovl.run_training_cycle(FORMATTED_TRAINING_DATA, VALID_DATA, epochs=1)
        print("Training completed successfully.")
    except Exception as e:
        print("Error during Training Stress Test:", e)
    
    print("\n-- Memory Test --")
    try:
        # Generate repeatedly with a large input to stress memory
        for i in range(5):
            result = sovl.generate("Test" * 50, max_new_tokens=100)
            print(f"Iteration {i + 1} completed.")
    except Exception as e:
        print("Error during Memory Test:", e)
    
    print("\n-- Error Recovery Test --")
    try:
        # Force a potential error with extreme input
        result = sovl.generate("", max_new_tokens=500)
        print("Generated:", result)
    except Exception as e:
        print("Error during Error Recovery Test:", e)

if __name__ == "__main__":
    stress_test()
