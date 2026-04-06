"""FastAPI backend for MyFaceDetect.

Run with: uvicorn api:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from pathlib import Path
import json
import uuid
from typing import List, Dict, Optional
import io

try:
    from myfacedetect import detect_faces
    HAS_MYFACEDETECT = True
except ImportError:
    HAS_MYFACEDETECT = False

app = FastAPI(
    title="MyFaceDetect API",
    description="Face detection and recognition API",
    version="0.4.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for background tasks
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "name": "MyFaceDetect API",
        "version": "0.3.0",
        "endpoints": [
            "POST /detect - Detect faces in image",
            "POST /detect-batch - Detect faces in multiple images",
            "POST /detect-url - Detect faces from URL",
            "GET /models - List available models",
            "GET /health - Health check"
        ]
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "myfacedetect_available": HAS_MYFACEDETECT
    }


@app.get("/models")
def list_models():
    """List available detection models."""
    return {
        "available_models": [
            "mediapipe",
            "haar",
            "ensemble"
        ],
        "settings": {
            "max_upload_size_mb": 50,
            "supported_formats": ["jpg", "jpeg", "png", "bmp"]
        }
    }


@app.post("/detect")
async def detect_faces_endpoint(
    file: UploadFile = File(...),
    method: str = "mediapipe",
    confidence_threshold: float = 0.5
):
    """Detect faces in uploaded image.
    
    Args:
        file: Image file
        method: Detection method (mediapipe, haar, ensemble)
        confidence_threshold: Minimum confidence score
        
    Returns:
        JSON with detected faces
    """
    if not HAS_MYFACEDETECT:
        raise HTTPException(status_code=500, detail="MyFaceDetect not available")
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file type
    allowed_types = {".jpg", ".jpeg", ".png", ".bmp"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Detect faces
        faces = detect_faces(image, method=method)
        
        # Filter by confidence
        filtered_faces = [
            f for f in faces 
            if f.get('confidence', 1.0) >= confidence_threshold
        ]
        
        return {
            "filename": file.filename,
            "image_shape": list(image.shape),
            "faces_detected": len(filtered_faces),
            "faces": [
                {
                    "id": i,
                    "bbox": {
                        "x": int(f['bbox'][0]),
                        "y": int(f['bbox'][1]),
                        "w": int(f['bbox'][2]),
                        "h": int(f['bbox'][3])
                    },
                    "confidence": float(f.get('confidence', 1.0)),
                    "landmarks": f.get('landmarks', [])
                }
                for i, f in enumerate(filtered_faces)
            ],
            "method": method
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect-batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    method: str = "mediapipe",
    confidence_threshold: float = 0.5
):
    """Detect faces in multiple images.
    
    Args:
        files: List of image files
        method: Detection method
        confidence_threshold: Minimum confidence score
        
    Returns:
        JSON with detection results for each image
    """
    if not HAS_MYFACEDETECT:
        raise HTTPException(status_code=500, detail="MyFaceDetect not available")
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": "Failed to decode"
                })
                continue
            
            faces = detect_faces(image, method=method)
            filtered_faces = [
                f for f in faces 
                if f.get('confidence', 1.0) >= confidence_threshold
            ]
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "faces_detected": len(filtered_faces),
                "faces": [
                    {
                        "id": i,
                        "bbox": {
                            "x": int(f['bbox'][0]),
                            "y": int(f['bbox'][1]),
                            "w": int(f['bbox'][2]),
                            "h": int(f['bbox'][3])
                        },
                        "confidence": float(f.get('confidence', 1.0))
                    }
                    for i, f in enumerate(filtered_faces)
                ]
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    total_faces = sum(r.get('faces_detected', 0) for r in results if r['status'] == 'success')
    
    return {
        "total_files": len(files),
        "successful": sum(1 for r in results if r['status'] == 'success'),
        "failed": sum(1 for r in results if r['status'] == 'error'),
        "total_faces": total_faces,
        "results": results,
        "method": method
    }


@app.post("/detect-url")
async def detect_from_url(
    image_url: str,
    method: str = "mediapipe",
    confidence_threshold: float = 0.5
):
    """Detect faces in image from URL.
    
    Args:
        image_url: URL to image
        method: Detection method
        confidence_threshold: Minimum confidence score
        
    Returns:
        JSON with detected faces
    """
    if not HAS_MYFACEDETECT:
        raise HTTPException(status_code=500, detail="MyFaceDetect not available")
    
    try:
        import urllib.request
        
        with urllib.request.urlopen(image_url) as response:
            image_data = response.read()
        
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image from URL")
        
        faces = detect_faces(image, method=method)
        filtered_faces = [
            f for f in faces 
            if f.get('confidence', 1.0) >= confidence_threshold
        ]
        
        return {
            "url": image_url,
            "image_shape": list(image.shape),
            "faces_detected": len(filtered_faces),
            "faces": [
                {
                    "id": i,
                    "bbox": {
                        "x": int(f['bbox'][0]),
                        "y": int(f['bbox'][1]),
                        "w": int(f['bbox'][2]),
                        "h": int(f['bbox'][3])
                    },
                    "confidence": float(f.get('confidence', 1.0))
                }
                for i, f in enumerate(filtered_faces)
            ],
            "method": method
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
