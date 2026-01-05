import cv2
import numpy as np
from ultralytics import YOLO
import json
import re
from typing import Optional, Tuple, List
import torch
import os

# Fix for Pillow 10.0+ compatibility with EasyOCR
try:
    from PIL import Image
    if not hasattr(Image, 'ANTIALIAS'):
        # ANTIALIAS was removed in Pillow 10.0, replaced with LANCZOS
        Image.ANTIALIAS = Image.LANCZOS
        print("✓ Applied Pillow 10.0+ compatibility patch")
except Exception as e:
    print(f"⚠ Could not apply Pillow compatibility patch: {e}")

import easyocr

# #region agent log
LOG_PATH = r"c:\Users\maazs\Documents\Projects\ALPR_TollPlaza_System\.cursor\debug.log"
def _log(location, message, data=None, hypothesisId=None):
    try:
        import json as _json
        log_entry = {"sessionId": "debug-session", "runId": "run1", "location": location, "message": message, "data": data or {}, "timestamp": int(__import__("time").time() * 1000)}
        if hypothesisId: log_entry["hypothesisId"] = hypothesisId
        with open(LOG_PATH, "a", encoding="utf-8") as f: f.write(_json.dumps(log_entry) + "\n")
    except: pass
# #endregion

class ALPREngine:
    def __init__(self, config_path: str = "config.json"):
        """Initialize ALPR engine with YOLO and EasyOCR"""
        # #region agent log
        _log("alpr_engine.py:13", "Loading config for ALPR", {"config_path": config_path}, "A")
        # #endregion
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # #region agent log
            _log("alpr_engine.py:14", "Config loaded", {"has_yolo": "yolo" in config, "has_ocr": "ocr" in config}, "B")
            # #endregion
        except FileNotFoundError as e:
            # #region agent log
            _log("alpr_engine.py:13", "Config file not found", {"error": str(e)}, "A")
            # #endregion
            raise
        except json.JSONDecodeError as e:
            # #region agent log
            _log("alpr_engine.py:13", "Config JSON decode error", {"error": str(e)}, "A")
            # #endregion
            raise
        except KeyError as e:
            # #region agent log
            _log("alpr_engine.py:16", "Missing config key", {"error": str(e)}, "B")
            # #endregion
            raise
        
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
        # #region agent log
        _log("alpr_engine.py:91", "Before preprocess_plate", {"plate_img_shape": plate_img.shape if plate_img is not None else None, "plate_img_is_none": plate_img is None}, "D")
        # #endregion
        if plate_img is None or plate_img.size == 0:
            # #region agent log
            _log("alpr_engine.py:91", "Invalid plate image", {}, "D")
            # #endregion
            raise ValueError("Plate image is None or empty")
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
    
    def read_plate_text(self, plate_img: np.ndarray) -> Optional[Tuple[str, float, np.ndarray]]:
        """
        Extract text from plate image using EasyOCR
        Returns: (plate_text, confidence, preprocessed_image) or None
        """
        # #region agent log
        _log("alpr_engine.py:read_plate_text:1", "read_plate_text() called", {"plate_img_shape": plate_img.shape if plate_img is not None else None}, "M")
        # #endregion
        
        # Preprocess image
        processed = self.preprocess_plate(plate_img)
        # #region agent log
        _log("alpr_engine.py:read_plate_text:2", "After preprocess_plate()", {"processed_shape": processed.shape if processed is not None else None}, "M")
        # #endregion
        
        # Run OCR
        # #region agent log
        _log("alpr_engine.py:read_plate_text:3", "Before ocr_reader.readtext()", {}, "M")
        # #endregion
        results = self.ocr_reader.readtext(processed)
        # #region agent log
        _log("alpr_engine.py:read_plate_text:4", "After ocr_reader.readtext()", {"results_count": len(results), "results": str(results)[:200]}, "M")
        # #endregion
        
        if not results:
            # #region agent log
            _log("alpr_engine.py:read_plate_text:5", "No OCR results", {}, "M")
            # #endregion
            return None
        
        # Get result with highest confidence
        best_result = max(results, key=lambda x: x[2])
        text = best_result[1]
        confidence = best_result[2]
        # #region agent log
        _log("alpr_engine.py:read_plate_text:6", "Best OCR result", {"raw_text": text, "confidence": confidence}, "M")
        # #endregion
        
        # Clean and format text
        text = self._clean_plate_text(text)
        # #region agent log
        _log("alpr_engine.py:read_plate_text:7", "After cleaning", {"cleaned_text": text, "min_confidence": self.ocr_config['confidence']}, "M")
        # #endregion
        
        if confidence < self.ocr_config['confidence']:
            # #region agent log
            _log("alpr_engine.py:read_plate_text:8", "Confidence too low, rejecting", {"confidence": confidence, "threshold": self.ocr_config['confidence']}, "M")
            # #endregion
            return None
        
        return text, confidence, processed
    
    def _clean_plate_text(self, text: str) -> str:
        """Clean and format plate text with Indian license plate aware corrections"""
        # Remove spaces and special characters
        text = re.sub(r'[^A-Z0-9\-]', '', text.upper())
        
        # Indian state codes for validation
        indian_state_codes = {
            'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'GA', 'GJ', 
            'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 
            'MN', 'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 'SK', 
            'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
        }
        
        # Fix common OCR misreadings in state codes (first 2 characters)
        if len(text) >= 2:
            first_two = text[:2]
            
            # Common misreadings for Indian state codes
            state_corrections = {
                'TH': 'TN',  # Tamil Nadu misread
                'HH': 'MH',  # Maharashtra misread  
                'KH': 'KA',  # Karnataka misread
                'PH': 'PB',  # Punjab misread
                'WH': 'WB',  # West Bengal misread
                'RH': 'RJ',  # Rajasthan misread
                'HN': 'MN',  # Manipur misread
                'HM': 'MM',  # Myanmar (though rare)
                'NH': 'MH',  # Another Maharashtra misread
                'TM': 'TN',  # Another Tamil Nadu misread
            }
            
            if first_two in state_corrections:
                text = state_corrections[first_two] + text[2:]
                print(f"✓ State code corrected: {first_two} → {state_corrections[first_two]}")
            elif first_two not in indian_state_codes:
                # Try to fix if it looks like a misread
                # H at position 0 or 1 is likely M or N
                if first_two[0] == 'H':
                    # Try common replacements
                    for replacement in ['M', 'N', 'K', 'W', 'R']:
                        candidate = replacement + first_two[1]
                        if candidate in indian_state_codes:
                            text = candidate + text[2:]
                            print(f"✓ State code corrected: {first_two} → {candidate}")
                            break
                elif first_two[1] == 'H':
                    # Second character is H, might be N
                    for replacement in ['N', 'M', 'A', 'P', 'J', 'R', 'K', 'L']:
                        candidate = first_two[0] + replacement
                        if candidate in indian_state_codes:
                            text = candidate + text[2:]
                            print(f"✓ State code corrected: {first_two} → {candidate}")
                            break
        
        # Common OCR corrections for numbers (but avoid in letter positions)
        # Indian format is typically: XX-00-XX-0000 or XX00XX0000
        # Letters appear at positions: 0-1 (state), and middle section
        
        # Apply number corrections more carefully
        result = []
        for i, char in enumerate(text):
            # First two characters should be letters (state code)
            if i < 2:
                # Keep as letter, but fix common issues
                if char == '0':
                    result.append('O')
                elif char == '1':
                    result.append('I')
                elif char == '5':
                    result.append('S')
                else:
                    result.append(char)
            # Characters 2-3 are typically numbers (district code)
            elif i >= 2 and i < 4:
                # Convert letters to numbers
                if char == 'O':
                    result.append('0')
                elif char == 'I':
                    result.append('1')
                elif char == 'S':
                    result.append('5')
                elif char == 'Z':
                    result.append('2')
                else:
                    result.append(char)
            # Middle section can be letters (series)
            elif i >= 4 and i < 8:
                # Could be letters or numbers, apply conservative corrections
                if char == 'O' and (i >= 6 or text[i-1].isdigit()):
                    result.append('0')
                elif char == 'I' and (i >= 6 or text[i-1].isdigit()):
                    result.append('1')
                else:
                    result.append(char)
            # End section is typically numbers
            else:
                # Convert letters to numbers
                if char == 'O':
                    result.append('0')
                elif char == 'I':
                    result.append('1')
                elif char == 'S':
                    result.append('5')
                elif char == 'Z':
                    result.append('2')
                else:
                    result.append(char)
        
        text = ''.join(result)
        
        return text
    
    def process_frame(self, frame: np.ndarray) -> Optional[dict]:
        """
        Complete ALPR pipeline: detect plate and read text
        Returns: dict with plate info or None
        """
        # #region agent log
        _log("alpr_engine.py:process_frame:1", "process_frame() called", {"frame_shape": frame.shape if frame is not None else None}, "M")
        # #endregion
        
        # Detect plate
        detection = self.detect_plate(frame)
        # #region agent log
        _log("alpr_engine.py:process_frame:2", "After detect_plate()", {"detection_is_none": detection is None}, "M")
        # #endregion
        if detection is None:
            return None
        
        plate_img, det_confidence = detection
        # #region agent log
        _log("alpr_engine.py:process_frame:3", "Plate detected", {"det_confidence": det_confidence, "plate_img_shape": plate_img.shape if plate_img is not None else None}, "M")
        # #endregion
        
        # Read text
        # #region agent log
        _log("alpr_engine.py:process_frame:4", "Before read_plate_text()", {}, "M")
        # #endregion
        ocr_result = self.read_plate_text(plate_img)
        # #region agent log
        _log("alpr_engine.py:process_frame:5", "After read_plate_text()", {"ocr_result_is_none": ocr_result is None}, "M")
        # #endregion
        if ocr_result is None:
            return None
        
        plate_text, ocr_confidence, preprocessed_img = ocr_result
        # #region agent log
        _log("alpr_engine.py:process_frame:6", "OCR succeeded", {"plate_text": plate_text, "ocr_confidence": ocr_confidence}, "M")
        # #endregion
        
        # Combined confidence
        combined_confidence = (det_confidence + ocr_confidence) / 2
        
        # #region agent log
        _log("alpr_engine.py:process_frame:7", "Returning result", {"plate_number": plate_text, "combined_confidence": combined_confidence}, "M")
        # #endregion
        
        return {
            'plate_number': plate_text,
            'confidence': combined_confidence,
            'detection_confidence': det_confidence,
            'ocr_confidence': ocr_confidence,
            'plate_image': plate_img,
            'preprocessed_image': preprocessed_img
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
