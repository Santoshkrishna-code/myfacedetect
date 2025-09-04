#!/usr/bin/env python3
"""
Run All Examples Script
======================

This script runs all available examples to demonstrate MyFaceDetect capabilities.
"""

import sys
import subprocess
from pathlib import Path
import time


def run_example(script_name, description):
    """Run an example script with error handling."""
    print(f"\n🔍 Running: {description}")
    print("-" * 50)

    script_path = Path("examples") / script_name

    if not script_path.exists():
        print(f"❌ Example not found: {script_path}")
        return False

    try:
        result = subprocess.run([sys.executable, str(script_path)],
                                capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print(result.stdout)
            print("✅ Example completed successfully!")
            return True
        else:
            print("❌ Example failed:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("⏰ Example timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running example: {e}")
        return False


def main():
    """Run all examples."""
    print("🚀 MyFaceDetect - Run All Examples")
    print("=" * 40)
    print("This script will run all available examples to demonstrate the library.")
    print("Note: Real-time examples will be skipped in batch mode.")
    print()

    examples = [
        ("basic_detection.py", "Basic Face Detection"),
        ("face_recognition.py", "Face Recognition with Training"),
        # Skip real-time example in batch mode
        # ("realtime_detection.py", "Real-time Detection (Webcam)")
    ]

    results = []

    for script, description in examples:
        success = run_example(script, description)
        results.append((description, success))

        if success:
            time.sleep(1)  # Brief pause between examples

    # Summary
    print("\n" + "=" * 50)
    print("📊 EXAMPLES SUMMARY:")
    print("=" * 50)

    successful = 0
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {description}")
        if success:
            successful += 1

    print(f"\nTotal: {successful}/{len(results)} examples passed")

    if successful == len(results):
        print("🎉 All examples ran successfully!")
    else:
        print("⚠️  Some examples failed - check error messages above")

    print("\n💡 To run real-time examples manually:")
    print("   python examples/realtime_detection.py")


if __name__ == "__main__":
    main()
