import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import json
import re
from typing import Optional, Tuple, List
import torch

class ALPREngine:
    def __init__(self, config_path: str = "config.json"):
        """Initialize ALPR engine with YOLO and EasyOCR"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.yolo_config = config['yolo']
        self.ocr_config = config['ocr']
        
        # Initialize YOLO model
        print("Loading YOLO model...")
        self.yolo_model = self._load_yolo_model()
        
        # Initialize EasyOCR
        print("Loading EasyOCR...")
        self.ocr_reader = easyocr.Reader(
            self.ocr_config['languages'],
            gpu=self.ocr_config['gpu']
        )
        
        print("✓ ALPR Engine initialized")
    
    def _load_yolo_model(self):
        """Load YOLO model for license plate detection"""
        try:
            # Try to load custom model
            model = YOLO(self.yolo_config['model_path'])
            print(f"✓ Loaded custom YOLO model: {self.yolo_config['model_path']}")
        except:
            # Fallback to YOLOv8n for general object detection
            print("Custom model not found, using YOLOv8n...")
            model = YOLO('yolov8n.pt')
            print("✓ Loaded YOLOv8n model")
        
        # Set device
        device = self.yolo_config['device']
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        model.to(device)
        print(f"✓ Using device: {device}")
        
        return model
    
    def detect_plate(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        Detect license plate in frame using YOLO
        Returns: (cropped_plate_image, confidence) or None
        """
        # Run YOLO detection
        results = self.yolo_model(frame, conf=self.yolo_config['confidence'])
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        # Get the detection with highest confidence
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        
        # Extract bounding box
        box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        confidence = float(confidences[best_idx])
        
        # Crop plate region with some padding
        h, w = frame.shape[:2]
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        plate_img = frame[y1:y2, x1:x2]
        
        return plate_img, confidence
    
    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """Preprocess plate image for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Resize for better OCR (height = 100px)
        height = 100
        ratio = height / thresh.shape[0]
        width = int(thresh.shape[1] * ratio)
        resized = cv2.resize(thresh, (width, height))
        
        return resized
    
    def read_plate_text(self, plate_img: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Extract text from plate image using EasyOCR
        Returns: (plate_text, confidence) or None
        """
        # Preprocess image
        processed = self.preprocess_plate(plate_img)
        
        # Run OCR
        results = self.ocr_reader.readtext(processed)
        
        if not results:
            return None
        
        # Get result with highest confidence
        best_result = max(results, key=lambda x: x[2])
        text = best_result[1]
        confidence = best_result[2]
        
        # Clean and format text
        text = self._clean_plate_text(text)
        
        if confidence < self.ocr_config['confidence']:
            return None
        
        return text, confidence
    
    def _clean_plate_text(self, text: str) -> str:
        """Clean and format plate text"""
        # Remove spaces and special characters
        text = re.sub(r'[^A-Z0-9\-]', '', text.upper())
        
        # Common OCR corrections
        text = text.replace('O', '0')  # O to 0
        text = text.replace('I', '1')  # I to 1
        text = text.replace('S', '5')  # S to 5
        text = text.replace('Z', '2')  # Z to 2
        
        return text
    
    def process_frame(self, frame: np.ndarray) -> Optional[dict]:
        """
        Complete ALPR pipeline: detect plate and read text
        Returns: dict with plate info or None
        """
        # Detect plate
        detection = self.detect_plate(frame)
        if detection is None:
            return None
        
        plate_img, det_confidence = detection
        
        # Read text
        ocr_result = self.read_plate_text(plate_img)
        if ocr_result is None:
            return None
        
        plate_text, ocr_confidence = ocr_result
        
        # Combined confidence
        combined_confidence = (det_confidence + ocr_confidence) / 2
        
        return {
            'plate_number': plate_text,
            'confidence': combined_confidence,
            'detection_confidence': det_confidence,
            'ocr_confidence': ocr_confidence,
            'plate_image': plate_img
        }
    
    def draw_detection(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw detection results on frame"""
        frame_copy = frame.copy()
        
        # Add text overlay
        text = f"{result['plate_number']} ({result['confidence']:.2f})"
        cv2.putText(
            frame_copy, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        return frame_copy
