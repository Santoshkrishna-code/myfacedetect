# Contributing to MyFaceDetect  

🎉 Thank you for your interest in contributing to **MyFaceDetect**!  
We're building a simple and powerful face detection library, and contributions of all kinds are welcome—whether it's bug fixes, documentation improvements, feature requests, or code contributions.  

---

## 📌 Ways to Contribute  

- 🐛 **Report Issues** – If you find a bug, please [open an issue](../../issues) with details and reproduction steps.  
- 💡 **Suggest Features** – Have an idea for an improvement? Share it in [Discussions](../../discussions) or create a feature request.  
- 🔧 **Code Contributions** – Submit PRs for bug fixes, new features, or performance improvements.  
- 📖 **Documentation** – Help us improve guides, examples, and explanations.  

---

## 🛠️ Development Setup  

Follow these steps to set up your local development environment:  

### 1. Fork and Clone

**Fork the repository** and clone your fork:  
```bash
git clone https://github.com/YOUR-USERNAME/myfacedetect.git
cd myfacedetect
```

**Original repository:**
```bash
git clone https://github.com/SantoshKrishna-code/myfacedetect.git
cd myfacedetect
```

### 2. Create Virtual Environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies

```bash
# Install runtime dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install in editable mode (so changes reflect immediately)
pip install -e .
```

---

## ▶️ Quick Testing

After installation, you can quickly test the package:

```python
from myfacedetect import detect_faces, detect_faces_realtime

# Test with a sample image (make sure you have an image file)
faces = detect_faces("sample.jpg", method="mediapipe")
print(f"Found {len(faces)} faces: {faces}")

# Test real-time detection with webcam (optional)
# detect_faces_realtime(method="both")
```

**Quick verification script:**
```python
# Create a simple test
import myfacedetect
import numpy as np
import cv2

# Create a test image
img = np.ones((200, 300, 3), dtype=np.uint8) * 255
cv2.imwrite('test.jpg', img)

# Test detection
faces = myfacedetect.detect_faces('test.jpg')
print(f"✅ Library working! Found {len(faces)} faces")

# Test FaceDetectionResult
result = myfacedetect.FaceDetectionResult((10, 20, 30, 40), 0.95, 'test')
print(f"✅ FaceDetectionResult: {result}")
```

---

## ✅ Running Tests

We use **pytest** for testing. Run the tests with:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=myfacedetect

# Run specific test file
pytest tests/test_core.py

# Run with verbose output
pytest -v
```

If you add new features, please include corresponding tests in the `tests/` directory.

---

## 🎨 Code Style

We follow **PEP8** for Python code style and use automated tools for consistency.

### Before committing, please run:

```bash
# Format code with black
black myfacedetect/

# Sort imports
isort myfacedetect/

# Check style with flake8
flake8 myfacedetect/
```

### Code Style Guidelines:

- **Line length**: Maximum 88 characters (black default)
- **Imports**: Use isort for consistent import ordering
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Encouraged for new code
- **Comments**: Clear, concise, and helpful

**Example of good code style:**
```python
from typing import List, Tuple, Optional
import cv2
import numpy as np

def detect_faces(
    image_path: str, 
    method: str = "mediapipe",
    min_confidence: float = 0.5
) -> List[FaceDetectionResult]:
    """Detect faces in an image using specified method.
    
    Args:
        image_path: Path to the image file
        method: Detection method ('haar', 'mediapipe', or 'both')
        min_confidence: Minimum confidence threshold
        
    Returns:
        List of FaceDetectionResult objects
        
    Raises:
        FileNotFoundError: If image file doesn't exist
    """
    # Implementation here
    pass
```

---

## 💻 Making a Contribution

### 1. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description

# Or for documentation
git checkout -b docs/update-readme
```

### 2. Branch Naming Convention

Use these prefixes for consistent branch naming:

- `feature/` - New features or enhancements
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Test additions or improvements
- `refactor/` - Code refactoring
- `perf/` - Performance improvements

### 3. Make Your Changes

- Write clean, readable code
- Add tests for new functionality
- Update documentation if needed
- Ensure all tests pass

### 4. Commit Your Changes

Use **Conventional Commit** messages:

```bash
# Feature commits
git commit -m "feat: add Haar Cascade support for face detection"

# Bug fix commits
git commit -m "fix: resolve TypeError in FaceDetectionResult.__repr__"

# Documentation commits
git commit -m "docs: update installation instructions in README"

# Test commits
git commit -m "test: add unit tests for MediaPipe detection"
```

### 5. Push and Create PR

```bash
# Push your branch
git push origin feature/your-feature-name

# Then create a Pull Request on GitHub
```

---

## 📜 Pull Request Guidelines

### Before submitting:

- ✅ **Keep PRs focused** - One feature/fix per PR
- ✅ **Write clear descriptions** - Explain what and why
- ✅ **Add tests** - Cover new functionality
- ✅ **Update docs** - Keep documentation current
- ✅ **Check CI** - Ensure all checks pass
- ✅ **Follow code style** - Run formatting tools

### PR Template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Added unit tests
- [ ] Manual testing completed
- [ ] All existing tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly marked)
```

---

## 🐛 Reporting Issues

When reporting bugs, please include:

### Issue Template:
```markdown
**Bug Description:**
Clear description of the issue

**To Reproduce:**
1. Step 1
2. Step 2
3. Error occurs

**Expected Behavior:**
What should happen

**Environment:**
- OS: [Windows 10/Ubuntu 20.04/macOS 12]
- Python: [3.8/3.9/3.10/3.11/3.12]
- MyFaceDetect version: [0.2.2]

**Additional Context:**
Screenshots, error logs, etc.
```

---

## 🔍 Project Structure

Understanding the codebase:

```
myfacedetect/
├── myfacedetect/          # Main package
│   ├── __init__.py        # Package initialization
│   ├── core.py            # Core detection functions
│   └── utils.py           # Utility functions (if any)
├── tests/                 # Test directory
│   ├── test_core.py       # Core functionality tests
│   └── test_utils.py      # Utility tests
├── requirements.txt       # Runtime dependencies
├── requirements-dev.txt   # Development dependencies
├── pyproject.toml        # Package configuration
├── README.md             # Project documentation
└── CONTRIBUTING.md       # This file
```

---

## 🎯 Development Tips

### Useful Commands:

```bash
# Install package in development mode
pip install -e .

# Run specific test
pytest tests/test_core.py::test_detect_faces

# Check test coverage
pytest --cov=myfacedetect --cov-report=html

# Format and check code
black . && isort . && flake8

# Build package locally
python -m build
```

### Testing Best Practices:

- Write tests for both success and error cases
- Use descriptive test names: `test_detect_faces_with_valid_image()`
- Mock external dependencies (cameras, file I/O) when appropriate
- Test edge cases and boundary conditions

---

## 📖 Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
Please treat others with respect and contribute to a welcoming environment.

---

## 🙌 Getting Help

If you're stuck or have questions:

- 📖 Check the [README](README.md) for usage examples
- 💬 Open a [Discussion](../../discussions) for general questions  
- 🐛 Create an [Issue](../../issues) for bug reports
- 📧 Contact maintainer: **B Santosh Krishna** at santoshkrishnabandla@gmail.com

---

## 🏆 Recognition

Contributors will be recognized in:

- **README.md** acknowledgments section
- **GitHub Contributors** page
- **Release notes** for significant contributions
- **CONTRIBUTORS.md** file (if we create one)

---

## � **Acknowledgments**

### Project Creator & Maintainer:
- **B Santosh Krishna** - Original author and primary maintainer
- GitHub: [@SantoshKrishna-code](https://github.com/SantoshKrishna-code)
- Email: santoshkrishnabandla@gmail.com

### Special Thanks:
- **OpenCV Team** - For the excellent computer vision library
- **Google MediaPipe Team** - For the powerful ML framework
- **Python Community** - For the amazing ecosystem
- **All Contributors** - Every bug report, feature request, and code contribution helps!

### Libraries & Dependencies:
- **OpenCV** (opencv-python) - Core computer vision functionality
- **MediaPipe** - Advanced face detection capabilities  
- **NumPy** - Numerical computing support
- **Pytest** - Testing framework

---

## �📚 Resources

### Learning Resources:
- [OpenCV Documentation](https://docs.opencv.org/)
- [MediaPipe Documentation](https://mediapipe.dev/)
- [Python Face Detection Tutorial](https://docs.opencv.org/master/db/d28/tutorial_cascade_classifier.html)

### Development Resources:
- [PEP8 Style Guide](https://pep8.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Pytest Documentation](https://docs.pytest.org/)

---

**Thanks for contributing to MyFaceDetect! 💜**

Every contribution, no matter how small, helps make this project better for everyone. We appreciate your time and effort in making MyFaceDetect more reliable, feature-rich, and accessible.

Happy coding! 🚀
