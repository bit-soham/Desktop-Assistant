#!/usr/bin/env python3
"""
Test script to validate all modules compile and import correctly.
Run this after making changes to any module files.
"""

import sys
import subprocess
import os

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Success!")
            return True
        else:
            print("✗ Failed!")
            print("Error output:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Error running command: {e}")
        return False

def main():
    print("Testing Python modules...")
    print("=" * 50)

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # List of modules to test
    modules = [
        "models/audio_processing.py",
        "models/text_processing.py",
        "models/note_management.py",
        "models/rag_llm.py",
        "models/model_setup.py",
        "models/utils.py"
    ]

    # Test 1: Syntax compilation
    print("\n1. Checking syntax compilation...")
    compile_cmd = f'python -m py_compile {" ".join(modules)}'
    if not run_command(compile_cmd, "Syntax compilation check"):
        return False

    # Test 2: Import test
    print("\n2. Testing imports...")
    import_cmd = 'python -c "import models.audio_processing; import models.text_processing; import models.note_management; import models.rag_llm; import models.model_setup; import models.utils; print(\'All imports successful!\')"'
    if not run_command(import_cmd, "Import validation"):
        return False

    print("\n" + "=" * 50)
    print("🎉 All tests passed! Your modules are ready.")
    print("You can now run: python main.py")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)