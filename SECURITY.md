# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions of MyFaceDetect:

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 0.2.x   | :white_check_mark: | Current stable release |
| 0.1.x   | :x:                | Legacy - please upgrade |
| < 0.1   | :x:                | Unsupported |

### Version Support Policy

- **Current Release (0.2.x)**: Full security support with immediate patches
- **Previous Major Version**: Security fixes for critical vulnerabilities only
- **Legacy Versions**: No security support - users should upgrade immediately

## Reporting a Vulnerability

We take security seriously and appreciate your efforts to responsibly disclose vulnerabilities.

### How to Report

1. **Email**: Send security reports to **santoshkrishnabandla@gmail.com**
2. **Subject Line**: Use "[SECURITY] MyFaceDetect Vulnerability Report"
3. **GitHub**: For non-critical issues, you can also create a private security advisory

### What to Include

Please provide the following information in your report:

- **Description**: Clear description of the vulnerability
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Impact Assessment**: Potential impact and severity
- **Affected Versions**: Which versions are affected
- **Proof of Concept**: Code or screenshots if applicable
- **Suggested Fix**: If you have ideas for remediation

### Response Timeline

- **Acknowledgment**: Within 48 hours of report submission
- **Initial Assessment**: Within 5 business days
- **Status Updates**: Weekly updates on investigation progress
- **Resolution**: Security fixes released as soon as possible, typically within 2-4 weeks

### What to Expect

#### If Vulnerability is Accepted:
- We'll work with you to understand and reproduce the issue
- A security patch will be developed and tested
- Credit will be given in the release notes (unless you prefer to remain anonymous)
- CVE assignment for critical vulnerabilities
- Coordinated disclosure timeline will be established

#### If Vulnerability is Declined:
- We'll provide a detailed explanation of why it's not considered a security issue
- Alternative solutions or mitigations may be suggested
- You're welcome to discuss the decision if you disagree

## Security Best Practices

When using MyFaceDetect in your applications, please follow these security guidelines:

### Input Validation
```python
import os
from pathlib import Path

# Validate file paths
def safe_image_path(image_path):
    path = Path(image_path).resolve()
    if not path.exists() or not path.is_file():
        raise ValueError("Invalid image path")
    return str(path)

# Use validated paths
faces = detect_faces(safe_image_path(user_input))
```

### Camera Access
```python
# Limit camera access in production
import cv2

def safe_camera_access(camera_index=0):
    if not isinstance(camera_index, int) or camera_index < 0:
        raise ValueError("Invalid camera index")
    return camera_index

# Use validated camera index
detect_faces_realtime(camera_index=safe_camera_access(user_camera_id))
```

### File Handling
- Always validate file paths and extensions
- Implement proper file size limits
- Use secure temporary directories
- Clean up temporary files after processing

### Network Security
- Never expose webcam feeds over unsecured connections
- Implement proper authentication for web applications
- Use HTTPS for any face detection web services
- Sanitize all user inputs

## Known Security Considerations

### Image Processing
- **Malicious Images**: Large or specially crafted images could cause memory exhaustion
- **Path Traversal**: Always validate and sanitize file paths
- **Resource Exhaustion**: Implement timeouts and resource limits

### Real-time Detection
- **Camera Access**: Ensure proper permissions and user consent
- **Privacy**: Be transparent about face detection usage
- **Data Storage**: Implement secure storage for captured images

### Dependencies
- We regularly update dependencies to address security vulnerabilities
- OpenCV and MediaPipe are actively maintained with security updates
- Monitor security advisories for all dependencies

## Security Updates

Security updates are distributed through:

1. **GitHub Releases**: All security patches are tagged and released
2. **PyPI**: Updated packages are immediately available via pip
3. **Security Advisories**: Critical vulnerabilities get dedicated advisories
4. **Changelog**: All security fixes are documented in CHANGELOG.md

## Contact Information

- **Security Email**: santoshkrishnabandla@gmail.com
- **GitHub Issues**: For non-security bugs and feature requests
- **Main Repository**: https://github.com/Santoshkrishna-code/myfacedetect

## Acknowledgments

We appreciate security researchers and users who help keep MyFaceDetect secure:

- Thank you to all users who responsibly report security issues
- Special recognition goes to contributors who help improve our security posture

---

**Last Updated**: August 27, 2025  
**Policy Version**: 1.0
