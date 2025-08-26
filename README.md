# MyFaceDetect

[![Python Version](https://img.shields.io/pypi/pyversions/myfacedetect.svg)](https://pypi.org/project/myfacedetect/)
[![License](https://img.shields.io/github/license/yourusername/myfacedetect.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/workflow/status/yourusername/myfacedetect/CI)](https://github.com/yourusername/myfacedetect/actions)

A comprehensive Python library for face detection in images and real-time video streams using OpenCV Haar cascades and MediaPipe.

## 🌟 Features

### Core Detection
- **Multiple Detection Methods**: OpenCV Haar cascades, MediaPipe, or both combined
- **Static Image Detection**: Process individual images with detailed results
- **Real-time Video Detection**: Live webcam detection with interactive controls
- **Batch Processing**: Efficiently process multiple images
- **Face Extraction**: Save individual face crops from images

### Advanced Features
- **Quality Analysis**: Analyze image quality metrics affecting detection
- **Benchmarking**: Compare performance of different detection methods
- **Result Export**: Export results to JSON, CSV formats
- **Visualization**: Create annotated images showing detection results
- **Configuration Management**: Customizable detection parameters
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

### Enhanced Real-time Detection
- **Interactive Controls**: Switch methods, capture screenshots, toggle settings
- **Performance Monitoring**: Real-time FPS display
- **Multiple Camera Support**: Support for different camera indices
- **Screenshot Capture**: Save detections with customizable output directory

## 🚀 Quick Start

### Installation

```bash
pip install myfacedetect
```

For development installation:
```bash
git clone https://github.com/yourusername/myfacedetect.git
cd myfacedetect
pip install -e .[dev]
```

### Basic Usage

```python
from myfacedetect import detect_faces, detect_faces_realtime

# Static image detection
faces = detect_faces("photo.jpg", method="mediapipe")
print(f"Found {len(faces)} faces")

for i, face in enumerate(faces):
    print(f"Face {i+1}: {face}")

# Real-time detection
detect_faces_realtime(method="both", show_fps=True)
```

### Advanced Usage

```python
from myfacedetect import detect_faces, batch_detect_faces
from myfacedetect.utils import create_detection_report, visualize_detection_results

# Advanced detection with visualization
faces, annotated_image = detect_faces(
    "photo.jpg", 
    method="both",
    return_image=True,
    scale_factor=1.05,  # More sensitive detection
    min_neighbors=3
)

# Create detailed report
report = create_detection_report(faces, "photo.jpg", "both", 0.123)

# Batch processing
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
all_results = batch_detect_faces(image_paths, method="mediapipe")

# Create visualization
visualization = visualize_detection_results(
    "photo.jpg", 
    faces, 
    "mediapipe",
    save_path="result.jpg"
)
```

## 📖 API Reference

### Core Functions

#### `detect_faces(image_path, method="mediapipe", **kwargs)`

Detect faces in an image with comprehensive options.

**Parameters:**
- `image_path` (str|Path|np.ndarray): Image file path or numpy array
- `method` (str): Detection method - "haar", "mediapipe", or "both"
- `return_image` (bool): Return annotated image with results
- `scale_factor` (float): Haar cascade scale factor (default: 1.1)
- `min_neighbors` (int): Haar cascade min neighbors (default: 4)
- `min_size` (tuple): Minimum face size (width, height) in pixels

**Returns:**
- List of `FaceDetectionResult` objects
- Optionally: tuple of (faces, annotated_image) if `return_image=True`

#### `detect_faces_realtime(camera_index=0, method="mediapipe", **kwargs)`

Real-time face detection with interactive controls.

**Parameters:**
- `camera_index` (int): Webcam index (default: 0)
- `method` (str): Detection method - "haar", "mediapipe", or "both"
- `window_name` (str): Display window name
- `show_fps` (bool): Display FPS counter
- `save_detections` (bool): Enable screenshot saving
- `output_dir` (str): Directory for saving screenshots

**Interactive Controls:**
- `ESC`: Exit detection
- `C` or `SPACE`: Capture screenshot
- `S`: Toggle screenshot saving
- `F`: Toggle FPS display
- `H`: Switch to Haar cascade method
- `M`: Switch to MediaPipe method
- `B`: Switch to both methods

### FaceDetectionResult Class

Represents a detected face with comprehensive information.

**Properties:**
- `bbox`: Bounding box as (x, y, width, height)
- `center`: Center point as (x, y)
- `confidence`: Detection confidence score (0.0-1.0)
- `x, y, width, height`: Individual bbox components

**Methods:**
- `__repr__()`: String representation with all details

### Utility Functions

#### `batch_detect_faces(image_paths, method="mediapipe", **kwargs)`
Process multiple images efficiently.

#### `save_face_crops(image_path, output_dir="face_crops", method="mediapipe")`
Extract and save individual face crops.

#### `benchmark_methods(image_paths, methods=["haar", "mediapipe"])`
Compare performance of different detection methods.

#### `create_detection_report(faces, image_path, method, execution_time)`
Generate detailed analysis report.

#### `visualize_detection_results(image_path, faces, method, save_path=None)`
Create annotated visualization of results.

## 🛠️ Configuration

MyFaceDetect supports configuration files for customizing detection parameters:

```python
from myfacedetect.config import config

# View current configuration
print(config.get("haar_cascade"))

# Modify parameters
config.set("mediapipe", "min_detection_confidence", 0.7)

# Save configuration
config.save_config()
```

Configuration file example (`myfacedetect_config.json`):
```json
{
  "haar_cascade": {
    "scale_factor": 1.05,
    "min_neighbors": 3,
    "min_size": [20, 20]
  },
  "mediapipe": {
    "min_detection_confidence": 0.7,
    "model_selection": 0
  }
}
```

## 🎮 Demo Script

Run the comprehensive demo:

```bash
# Interactive demo
python -m myfacedetect.demo

# Command line options
python -m myfacedetect.demo --image photo.jpg --method both
python -m myfacedetect.demo --realtime
python -m myfacedetect.demo --batch ./photos
python -m myfacedetect.demo --advanced photo.jpg
```

## 🔧 Development

### Setup Development Environment

```bash
git clone https://github.com/yourusername/myfacedetect.git
cd myfacedetect

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install in development mode
pip install -e .[dev]
```

### Run Tests

```bash
pytest tests/ -v --cov=myfacedetect
```

### Code Formatting

```bash
black myfacedetect/
isort myfacedetect/
flake8 myfacedetect/
```

## 📊 Performance Comparison

| Method | Speed | Accuracy | Resource Usage |
|--------|-------|----------|----------------|
| Haar Cascade | Fast | Good | Low |
| MediaPipe | Medium | Excellent | Medium |
| Both Combined | Slower | Best | Higher |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV team for the excellent computer vision library
- MediaPipe team for the powerful ML framework
- Contributors and users of this library

## 📚 Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [MediaPipe Documentation](https://mediapipe.dev/)
- [Face Detection Guide](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html)

---

**Made with ❤️ by [Your Name]**
