# Publishing MyFaceDetect to PyPI

## Version Information
- **Current Version**: 0.4.0
- **Package Name**: myfacedetect
- **PyPI URL**: https://pypi.org/project/myfacedetect/

## Prerequisites

1. PyPI Account
   - Create account at https://pypi.org/account/register/
   - Create Token at https://pypi.org/manage/account/

2. Build Tools
   ```bash
   pip install --upgrade build twine wheel
   ```

3. Configuration (`.pypirc`)
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = <your-token-here>

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = <your-test-token-here>
   ```

## Package Contents

### What's Included
- ✅ Complete face detection framework
- ✅ Recognition system with multiple embeddings
- ✅ Advanced training pipeline (PyTorch)
- ✅ Experiment tracking and hyperparameter optimization
- ✅ Security features (liveness detection, privacy protection)
- ✅ Comprehensive documentation
- ✅ 19+ unit tests
- ✅ Docker support

### Package Size
- **Wheel**: ~80 KB
- **Source**: ~170 KB
- **Installation Size**: ~500 MB (with dependencies)

## Building the Package

### Clean Build
```bash
# Remove old builds
rm -rf build/ dist/ myfacedetect.egg-info/

# Build new packages
python -m build
```

### Verify Package
```bash
# Check package structure
twine check dist/*

# List contents
unzip -l dist/myfacedetect-0.4.0-py3-none-any.whl
```

## Publishing Process

### Test PyPI (Recommended First)
```bash
# Upload to test PyPI
twine upload --repository testpypi dist/* -u __token__ -p "your-test-token"

# Install from test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ myfacedetect
```

### Production PyPI
```bash
# Upload to production PyPI
twine upload dist/* -u __token__ -p "your-token"

# Install from PyPI
pip install myfacedetect
```

## Installation Options

### Basic Installation
```bash
pip install myfacedetect
```

### With Training Framework
```bash
pip install myfacedetect[training]
```

### With All Features
```bash
pip install myfacedetect[all]
```

### Development Installation
```bash
pip install myfacedetect[dev]
```

## Version History

### v0.4.0 (Current)
- Advanced PyTorch training framework
- Bayesian hyperparameter optimization (Optuna)
- Experiment tracking and checkpoints
- 100% test coverage (19/19 tests passing)
- Enhanced documentation for training workflow
- Docker containerization support

### v0.3.0
- Modular detector architecture
- Face recognition system
- Security features (liveness detection)
- Privacy protection
- Performance optimization

## Support

- **Documentation**: https://github.com/Santoshkrishna-code/myfacedetect
- **Issues**: https://github.com/Santoshkrishna-code/myfacedetect/issues
- **Releases**: https://github.com/Santoshkrishna-code/myfacedetect/releases

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.
