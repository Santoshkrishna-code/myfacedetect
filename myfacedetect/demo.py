#!/usr/bin/env python3
"""
Enhanced demo script for MyFaceDetect library.
This script showcases all the advanced features of the library.
"""
import argparse
import sys
from pathlib import Path
import logging

from myfacedetect import detect_faces, detect_faces_realtime, FaceDetectionResult
from myfacedetect.utils import (
    benchmark_methods, create_face_dataset, analyze_image_quality,
    create_detection_report, visualize_detection_results, export_results
)
from myfacedetect.config import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_static_detection(image_path: str, method: str = "mediapipe"):
    """Demo static image detection with all features."""
    print(f"\n🖼️  Static Image Detection Demo - {method.upper()}")
    print("=" * 50)

    try:
        import time
        start_time = time.time()

        # Detect faces with different options
        faces, annotated_image = detect_faces(
            image_path,
            method=method,
            return_image=True
        )

        execution_time = time.time() - start_time

        print(f"✅ Detected {len(faces)} faces in {execution_time:.3f} seconds")

        # Show face details
        for i, face in enumerate(faces):
            print(f"  Face #{i+1}: {face}")

        # Create detailed report
        report = create_detection_report(
            faces, image_path, method, execution_time)

        # Show quality metrics
        print(f"\n📊 Image Quality Metrics:")
        for metric, value in report["quality_metrics"].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")

        # Save visualization
        output_path = Path("demo_detection.jpg")
        visualization = visualize_detection_results(
            image_path, faces, method, save_path=output_path
        )
        print(f"📁 Visualization saved to: {output_path}")

        return faces, report

    except Exception as e:
        print(f"❌ Error: {e}")
        return [], {}


def demo_realtime_detection():
    """Demo real-time detection with enhanced features."""
    print(f"\n🎥 Real-time Detection Demo")
    print("=" * 50)
    print("Controls:")
    print("  ESC: Exit")
    print("  'c' or SPACE: Capture screenshot")
    print("  's': Toggle screenshot saving")
    print("  'f': Toggle FPS display")
    print("  'h': Switch to Haar cascade")
    print("  'm': Switch to MediaPipe")
    print("  'b': Switch to both methods")

    try:
        detect_faces_realtime(
            method="mediapipe",
            show_fps=True,
            save_detections=True,
            output_dir="demo_captures"
        )
        print("✅ Real-time detection completed")

    except Exception as e:
        print(f"❌ Error: {e}")


def demo_batch_processing(image_dir: str):
    """Demo batch processing capabilities."""
    print(f"\n📁 Batch Processing Demo")
    print("=" * 50)

    image_path = Path(image_dir)
    if not image_path.exists():
        print(f"❌ Directory not found: {image_dir}")
        return

    # Find all image files
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in image_path.rglob('*')
                   if f.suffix.lower() in supported_formats]

    if not image_files:
        print(f"❌ No image files found in {image_dir}")
        return

    print(f"📊 Found {len(image_files)} images to process")

    # Benchmark different methods
    if len(image_files) > 0:
        sample_images = image_files[:min(5, len(image_files))]
        print(
            f"\n⏱️  Benchmarking methods on {len(sample_images)} sample images...")

        benchmark_results = benchmark_methods(
            sample_images,
            methods=["haar", "mediapipe"],
            iterations=2
        )

        for method, results in benchmark_results.items():
            print(f"\n{method.upper()} Results:")
            print(f"  Average time: {results['average_time']:.3f} seconds")
            print(f"  Total faces found: {results['total_faces']}")
            print(f"  Images processed: {results['images_processed']}")
            print(f"  Errors: {results['errors']}")

    # Create face dataset
    print(f"\n👥 Creating face dataset...")
    dataset_stats = create_face_dataset(
        image_dir,
        "demo_face_dataset",
        method="mediapipe",
        min_face_size=50
    )

    print(f"Dataset Statistics:")
    for key, value in dataset_stats.items():
        print(f"  {key}: {value}")


def demo_advanced_features(image_path: str):
    """Demo advanced analysis features."""
    print(f"\n🔬 Advanced Features Demo")
    print("=" * 50)

    # Quality analysis
    print("📈 Analyzing image quality...")
    quality_metrics = analyze_image_quality(image_path)

    print("Quality Metrics:")
    for metric, value in quality_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")

    # Detection with different methods
    methods = ["haar", "mediapipe", "both"]
    all_reports = []

    for method in methods:
        print(f"\n🔍 Testing {method.upper()} method...")
        faces, report = demo_static_detection(image_path, method)
        all_reports.append(report)

    # Export results
    if all_reports:
        print("\n📤 Exporting results...")
        export_results(all_reports, "demo_results.json", "json")
        export_results(all_reports, "demo_results.csv", "csv")
        print("✅ Results exported to demo_results.json and demo_results.csv")


def main():
    """Main demo function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="MyFaceDetect Library Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --realtime                    # Start real-time demo
  %(prog)s --image photo.jpg             # Detect faces in image
  %(prog)s --batch ./photos              # Process all images in directory
  %(prog)s --advanced photo.jpg          # Full analysis demo
  %(prog)s --config                      # Show configuration
        """
    )

    parser.add_argument('--image', '-i',
                        help='Path to image file for static detection')
    parser.add_argument('--realtime', '-r', action='store_true',
                        help='Start real-time face detection')
    parser.add_argument('--batch', '-b',
                        help='Directory path for batch processing')
    parser.add_argument('--advanced', '-a',
                        help='Path to image for advanced features demo')
    parser.add_argument('--config', '-c', action='store_true',
                        help='Show current configuration')
    parser.add_argument('--method', '-m',
                        choices=['haar', 'mediapipe', 'both'],
                        default='mediapipe',
                        help='Detection method to use')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("🎯 MyFaceDetect Library - Enhanced Demo")
    print("=" * 60)

    if args.config:
        print("⚙️  Current Configuration:")
        print("=" * 30)
        for section, values in config._config.items():
            print(f"{section}:")
            for key, value in values.items():
                print(f"  {key}: {value}")
            print()
        return

    if args.image:
        demo_static_detection(args.image, args.method)

    elif args.realtime:
        demo_realtime_detection()

    elif args.batch:
        demo_batch_processing(args.batch)

    elif args.advanced:
        demo_advanced_features(args.advanced)

    else:
        # Interactive mode
        print("🎮 Interactive Demo Mode")
        print("=" * 30)
        print("Choose an option:")
        print("1. 📸 Static image detection")
        print("2. 🎥 Real-time detection")
        print("3. 📁 Batch processing")
        print("4. 🔬 Advanced features")
        print("5. ⚙️  Show configuration")
        print("6. 🚪 Exit")

        while True:
            try:
                choice = input("\n👉 Enter choice (1-6): ").strip()

                if choice == '1':
                    image_path = input("📁 Enter image path: ").strip()
                    if image_path:
                        demo_static_detection(image_path, args.method)

                elif choice == '2':
                    demo_realtime_detection()

                elif choice == '3':
                    batch_dir = input("📁 Enter directory path: ").strip()
                    if batch_dir:
                        demo_batch_processing(batch_dir)

                elif choice == '4':
                    image_path = input("📁 Enter image path: ").strip()
                    if image_path:
                        demo_advanced_features(image_path)

                elif choice == '5':
                    print("\n⚙️  Current Configuration:")
                    for section, values in config._config.items():
                        print(f"{section}: {values}")

                elif choice == '6':
                    print("👋 Thank you for using MyFaceDetect!")
                    break

                else:
                    print("❌ Invalid choice. Please enter 1-6.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
