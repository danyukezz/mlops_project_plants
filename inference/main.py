"""
FastAPI Inference Service for Plant Disease Classification
"""
import os
import io
import json
from pathlib import Path
from typing import Optional
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch.nn as nn
from torchvision import models, transforms

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Recognition API",
    description="API for detecting plant diseases from images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Changed to False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None
class_names = []


class PlantDiseaseModel(nn.Module):
    """Plant Disease Classification Model"""
    
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    disease: str
    confidence: float
    class_index: int
    all_predictions: Optional[dict] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str


def load_model():
    """Load the trained model"""
    global model, device, class_names
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find model file - search recursively
    model_dir = Path("./model")
    model_files = list(model_dir.rglob("*.pth"))
    
    if not model_files:
        print(f"Warning: No model file found in {model_dir}")
        return False
    
    model_path = model_files[0]
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load class names - search recursively
    class_names_files = list(model_dir.rglob("class_names.json"))
    if class_names_files:
        with open(class_names_files[0], 'r') as f:
            class_names = json.load(f)
    else:
        class_names = checkpoint.get('class_names', [])
    
    num_classes = len(class_names)
    
    # Initialize model
    model = PlantDiseaseModel(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully with {num_classes} classes")
    return True


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("Loading model...")
    success = load_model()
    if not success:
        print("Warning: Model not loaded")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Plant Disease Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "classes": "/classes",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device) if device else "unknown"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), top_k: int = 5):
    """
    Predict plant disease from uploaded image
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top k predictions
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(class_names)))
            
            top_probs = top_probs[0].cpu().numpy()
            top_indices = top_indices[0].cpu().numpy()
        
        # Get top prediction
        top_disease = class_names[top_indices[0]]
        top_confidence = float(top_probs[0])
        
        # Create all predictions dict
        all_predictions = {
            class_names[idx]: float(prob)
            for idx, prob in zip(top_indices, top_probs)
        }
        
        return PredictionResponse(
            disease=top_disease,
            confidence=top_confidence,
            class_index=int(top_indices[0]),
            all_predictions=all_predictions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/classes")
async def get_classes():
    """Get list of all disease classes"""
    if not class_names:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "num_classes": len(class_names),
        "classes": class_names
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
