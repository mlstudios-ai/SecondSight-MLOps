from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel

router = APIRouter()

@router.get("/api/hazard-detection/detect")
async def detect():
    """Get detected hazard object list"""
    
    objects = [
        {
            "class_label": "pot_hole",
            "confidence_score": 65,
            "bounding_box": [23424,2323],
            "image_center": [232, 343],
            "prob_score": 45,
            "class_id": 2324254,
            "aspect_ratio": 0.4,
            "meta_data": [],
            "timestamp": 24324234
        },        
        {
            "class_label": "fire",
            "confidence_score": 65,
            "bounding_box": [23424,2323],
            "image_center": [232, 343],
            "prob_score": 45,
            "class_id": 2324254,
            "aspect_ratio": 0.4,
            "meta_data": [],
            "timestamp": 24324234
        }
    ]
   
    return JSONResponse(content=objects)

@router.get("/api/image-description/describe")
async def describe():
    """Fetch the current user from the client-side or server-side variable."""
    
    image_data = {
            description: "Test image description: dummy data",
            timestamp: 24324234
        }
    
    return JSONResponse(content=image_data)
