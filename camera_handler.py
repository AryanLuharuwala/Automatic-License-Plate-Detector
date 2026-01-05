import cv2
import numpy as np
from typing import Optional, Tuple
import json
from threading import Thread, Lock
import time
import os

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

class CameraHandler:
    def __init__(self, config_path: str = "config.json"):
        """Initialize camera handler"""
        # #region agent log
        _log("camera_handler.py:11", "Loading config for camera", {"config_path": config_path}, "A")
        # #endregion
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # #region agent log
            _log("camera_handler.py:12", "Config loaded", {"has_camera_key": "camera" in config}, "B")
            # #endregion
        except FileNotFoundError as e:
            # #region agent log
            _log("camera_handler.py:11", "Config file not found", {"error": str(e)}, "A")
            # #endregion
            raise
        except json.JSONDecodeError as e:
            # #region agent log
            _log("camera_handler.py:11", "Config JSON decode error", {"error": str(e)}, "A")
            # #endregion
            raise
        except KeyError as e:
            # #region agent log
            _log("camera_handler.py:14", "Missing camera config key", {"error": str(e)}, "B")
            # #endregion
            raise
        
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
        self.camera_available = False
        
        self.initialize_camera()
    
    def initialize_camera(self):
        """Initialize camera capture"""
        # #region agent log
        _log("camera_handler.py:init_cam:1", "Before cv2.VideoCapture()", {"camera_source": self.camera_source, "camera_source_type": type(self.camera_source).__name__}, "G")
        # #endregion
        try:
            # On Windows, use CAP_DSHOW backend for faster initialization
            import platform
            if platform.system() == 'Windows':
                # #region agent log
                _log("camera_handler.py:init_cam:1.5", "Using CAP_DSHOW backend on Windows", {}, "L")
                # #endregion
                self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.camera_source)
            # #region agent log
            _log("camera_handler.py:init_cam:2", "After cv2.VideoCapture()", {"cap_is_none": self.cap is None, "cap_is_opened": self.cap.isOpened() if self.cap else False}, "G")
            # #endregion
            
            if not self.cap.isOpened():
                # #region agent log
                _log("camera_handler.py:init_cam:3", "Camera not opened", {"camera_source": self.camera_source}, "J")
                # #endregion
                raise Exception(f"Cannot open camera: {self.camera_source}")
            
            # #region agent log
            _log("camera_handler.py:init_cam:4", "Before setting camera properties", {}, "H")
            # #endregion
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # #region agent log
            _log("camera_handler.py:init_cam:5", "After setting camera properties", {}, "H")
            # #endregion
            
            self.camera_available = True
            # #region agent log
            _log("camera_handler.py:init_cam:6", "Camera initialized successfully", {"width": self.width, "height": self.height, "fps": self.fps}, "G")
            # #endregion
            print(f"✓ Camera initialized: {self.camera_source}")
            print(f"  Resolution: {self.width}x{self.height} @ {self.fps}fps")
            
        except Exception as e:
            # #region agent log
            _log("camera_handler.py:init_cam:7", "Camera initialization exception", {"error": str(e), "error_type": type(e).__name__}, "K")
            # #endregion
            self.camera_available = False
            print(f"✗ Camera initialization failed: {e}")
            print("  GUI will continue without camera functionality")
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def start(self):
        """Start camera capture thread"""
        if not self.camera_available:
            print("⚠ Camera not available, skipping capture thread")
            return
        
        if self.running:
            return
        
        self.running = True
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("✓ Camera capture started")
    
    def _capture_loop(self):
        """Continuous frame capture loop"""
        if not self.cap or not self.camera_available:
            return
        while self.running:
            if self.cap:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame = frame
                time.sleep(1 / self.fps)
            else:
                break
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame"""
        # #region agent log
        _log("camera_handler.py:70", "Before get_frame", {"frame_is_none": self.frame is None}, "D")
        # #endregion
        with self.lock:
            if self.frame is None:
                # #region agent log
                _log("camera_handler.py:70", "Frame is None, returning None", {}, "D")
                # #endregion
                return None
            try:
                result = self.frame.copy()
                # #region agent log
                _log("camera_handler.py:70", "Frame copied successfully", {"result_shape": result.shape if result is not None else None}, "D")
                # #endregion
                return result
            except AttributeError as e:
                # #region agent log
                _log("camera_handler.py:70", "Frame copy failed", {"error": str(e), "frame_type": type(self.frame).__name__}, "D")
                # #endregion
                return None
    
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
            self.cap = None
        self.camera_available = False
        print("✓ Camera released")
    
    def is_opened(self) -> bool:
        """Check if camera is open"""
        return self.camera_available and self.cap is not None and self.cap.isOpened()
