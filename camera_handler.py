import cv2
import numpy as np
from typing import Optional, Tuple
import json
from threading import Thread, Lock
import time

class CameraHandler:
    def __init__(self, config_path: str = "config.json"):
        """Initialize camera handler"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.camera_config = config['camera']
        self.camera_source = self.camera_config['source']
        self.width = self.camera_config['width']
        self.height = self.camera_config['height']
        self.fps = self.camera_config['fps']
        
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = Lock()
        self.thread = None
        
        self.initialize_camera()
    
    def initialize_camera(self):
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            
            if not self.cap.isOpened():
                raise Exception(f"Cannot open camera: {self.camera_source}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            print(f"✓ Camera initialized: {self.camera_source}")
            print(f"  Resolution: {self.width}x{self.height} @ {self.fps}fps")
            
        except Exception as e:
            print(f"✗ Camera initialization failed: {e}")
            raise
    
    def start(self):
        """Start camera capture thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("✓ Camera capture started")
    
    def _capture_loop(self):
        """Continuous frame capture loop"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            time.sleep(1 / self.fps)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame (OpenCV compatible)"""
        frame = self.get_frame()
        return (frame is not None, frame)
    
    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.thread:
            self.thread.join()
        print("✓ Camera capture stopped")
    
    def release(self):
        """Release camera resources"""
        self.stop()
        if self.cap:
            self.cap.release()
        print("✓ Camera released")
    
    def is_opened(self) -> bool:
        """Check if camera is open"""
        return self.cap is not None and self.cap.isOpened()
