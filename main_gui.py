import sys
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTabWidget,
                             QTableWidget, QTableWidgetItem, QLineEdit, 
                             QComboBox, QTextEdit, QMessageBox, QHeaderView,
                             QFrame, QGroupBox, QScrollArea, QStackedWidget,
                             QSizePolicy, QSpacerItem, QFileDialog)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QIcon
import cv2
import numpy as np
from datetime import datetime
import os

from database import DatabaseManager
from camera_handler import CameraHandler
from alpr_engine import ALPREngine

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

class CameraInitThread(QThread):
    """Thread for initializing camera in background"""
    initialization_complete = pyqtSignal(object)  # Emits the CameraHandler object or None
    initialization_status = pyqtSignal(str)  # Emits status messages
    
    def run(self):
        # #region agent log
        _log("main_gui.py:CameraInitThread:1", "CameraInitThread.run() started", {}, "J")
        # #endregion
        try:
            self.initialization_status.emit("Initializing camera...")
            # #region agent log
            _log("main_gui.py:CameraInitThread:2", "Before importing CameraHandler", {}, "J")
            # #endregion
            from camera_handler import CameraHandler
            # #region agent log
            _log("main_gui.py:CameraInitThread:3", "Before creating CameraHandler()", {}, "J")
            # #endregion
            camera = CameraHandler()
            # #region agent log
            _log("main_gui.py:CameraInitThread:4", "After creating CameraHandler()", {"camera_available": camera.camera_available if camera else False}, "J")
            # #endregion
            if camera.camera_available:
                # #region agent log
                _log("main_gui.py:CameraInitThread:5", "Before camera.start()", {}, "J")
                # #endregion
                camera.start()
                # #region agent log
                _log("main_gui.py:CameraInitThread:6", "After camera.start()", {}, "J")
                # #endregion
                self.initialization_status.emit("Camera initialized successfully!")
            else:
                # #region agent log
                _log("main_gui.py:CameraInitThread:7", "Camera not available", {}, "J")
                # #endregion
                self.initialization_status.emit("Camera not available")
            # #region agent log
            _log("main_gui.py:CameraInitThread:8", "Before emitting initialization_complete", {"camera_is_none": camera is None}, "J")
            # #endregion
            self.initialization_complete.emit(camera)
            # #region agent log
            _log("main_gui.py:CameraInitThread:9", "After emitting initialization_complete", {}, "J")
            # #endregion
        except Exception as e:
            # #region agent log
            _log("main_gui.py:CameraInitThread:10", "Camera initialization exception in thread", {"error": str(e), "error_type": type(e).__name__}, "K")
            # #endregion
            self.initialization_status.emit(f"Camera initialization failed: {e}")
            self.initialization_complete.emit(None)


class ALPRInitThread(QThread):
    """Thread for initializing ALPR engine in background"""
    initialization_complete = pyqtSignal(object)  # Emits the ALPREngine object or None
    initialization_status = pyqtSignal(str)  # Emits status messages
    
    def run(self):
        try:
            self.initialization_status.emit("Loading ALPR Engine...")
            from alpr_engine import ALPREngine
            alpr_engine = ALPREngine()
            self.initialization_status.emit("ALPR Engine loaded successfully!")
            self.initialization_complete.emit(alpr_engine)
        except Exception as e:
            self.initialization_status.emit(f"ALPR Engine failed: {e}")
            self.initialization_complete.emit(None)


class DetectionThread(QThread):
    """Thread for continuous plate detection"""
    detection_result = pyqtSignal(dict)
    
    def __init__(self, camera, alpr_engine):
        super().__init__()
        self.camera = camera
        self.alpr_engine = alpr_engine
        self.running = False
        self.paused = False
    
    def run(self):
        self.running = True
        print("\n" + "="*60)
        print("🔍 DETECTION THREAD STARTED")
        print("="*60 + "\n")
        
        while self.running:
            if not self.paused and self.camera and self.camera.camera_available and self.alpr_engine:
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    try:
                        # #region agent log
                        _log("main_gui.py:DetectionThread:1", "Before process_frame()", {"frame_shape": frame.shape if frame is not None else None}, "M")
                        # #endregion
                        result = self.alpr_engine.process_frame(frame)
                        # #region agent log
                        _log("main_gui.py:DetectionThread:2", "After process_frame()", {"result_is_none": result is None, "result_keys": list(result.keys()) if result else None}, "M")
                        # #endregion
                        if result:
                            print("\n" + "🎯 "*20)
                            print(f"🎯 PLATE DETECTED: {result.get('plate_number')} (Confidence: {result.get('confidence'):.2%})")
                            print("🎯 "*20 + "\n")
                            print(f"📤 EMITTING DETECTION SIGNAL...")
                            
                            # Add the frame to the result for snapshot
                            result['frame'] = frame.copy()
                            
                            # Pause detection after finding a plate
                            self.paused = True
                            print(f"⏸️  DETECTION PAUSED")
                            
                            # #region agent log
                            _log("main_gui.py:DetectionThread:3", "Before emitting detection_result", {"plate_number": result.get('plate_number'), "confidence": result.get('confidence')}, "N")
                            # #endregion
                            self.detection_result.emit(result)
                            print(f"✅ SIGNAL EMITTED SUCCESSFULLY\n")
                            # #region agent log
                            _log("main_gui.py:DetectionThread:4", "After emitting detection_result", {}, "N")
                            # #endregion
                    except Exception as e:
                        print(f"\n❌ ERROR IN DETECTION THREAD: {e}")
                        print(f"   Error type: {type(e).__name__}\n")
                        # #region agent log
                        _log("main_gui.py:DetectionThread:5", "Exception in detection thread", {"error": str(e), "error_type": type(e).__name__}, "P")
                        # #endregion
            self.msleep(100)
    
    def pause(self):
        """Pause detection"""
        self.paused = True
        print("⏸️  Detection paused")
    
    def resume(self):
        """Resume detection"""
        self.paused = False
        print("▶️  Detection resumed")
    
    def stop(self):
        self.running = False


class MetricCard(QFrame):
    """Modern metric card widget"""
    def __init__(self, title, value, subtitle="", icon="", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }
        """)
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)
        self.setLayout(layout)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #6b7280; font-size: 14px; font-weight: 500;")
        layout.addWidget(title_label)
        
        # Value (store reference for updates)
        self.value_label = QLabel(str(value))
        self.value_label.setStyleSheet("color: #111827; font-size: 32px; font-weight: 700;")
        layout.addWidget(self.value_label)
        
        # Subtitle
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setStyleSheet("color: #9ca3af; font-size: 12px;")
            layout.addWidget(subtitle_label)
    
    def set_value(self, value):
        """Update the metric value"""
        self.value_label.setText(str(value))


class SidebarButton(QPushButton):
    """Sidebar navigation button"""
    def __init__(self, text, icon="", parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #6b7280;
                text-align: left;
                padding: 12px 20px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                border: none;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
                color: #111827;
            }
            QPushButton:checked {
                background-color: #3b82f6;
                color: #ffffff;
            }
        """)
        self.setMinimumHeight(44)


class ALPRMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # #region agent log
        _log("main_gui.py:__init__:1", "ALPRMainWindow.__init__ started", {}, "D")
        # #endregion
        self.setWindowTitle("ALPR Toll Plaza System")
        self.setGeometry(100, 50, 1920, 1080)
        self.setMinimumSize(1400, 800)
        
        # Load configuration
        # #region agent log
        _log("main_gui.py:48", "Loading config file", {"config_path": "config.json"}, "A")
        # #endregion
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
            # #region agent log
            _log("main_gui.py:49", "Config loaded successfully", {"has_database": "database" in self.config, "has_camera": "camera" in self.config, "has_yolo": "yolo" in self.config, "has_ocr": "ocr" in self.config, "has_node": "node" in self.config}, "B")
            # #endregion
        except FileNotFoundError as e:
            # #region agent log
            _log("main_gui.py:48", "Config file not found", {"error": str(e)}, "A")
            # #endregion
            QMessageBox.critical(self, "Config Error", f"Config file not found: {e}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            # #region agent log
            _log("main_gui.py:48", "Config JSON decode error", {"error": str(e)}, "A")
            # #endregion
            QMessageBox.critical(self, "Config Error", f"Invalid JSON in config file: {e}")
            sys.exit(1)
        except KeyError as e:
            # #region agent log
            _log("main_gui.py:48", "Missing config key", {"error": str(e), "config_keys": list(self.config.keys())}, "B")
            # #endregion
            QMessageBox.critical(self, "Config Error", f"Missing required config key: {e}")
            sys.exit(1)
        
        # Initialize components
        self.db = None
        self.camera = None
        self.alpr_engine = None
        self.camera_loading = False
        self.alpr_loading = False
        self.detection_thread = None
        
        # Initialize database (required)
        try:
            self.db = DatabaseManager()
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to initialize database: {e}")
            sys.exit(1)
        
        # Camera and ALPR will be initialized in background after GUI is shown
        # #region agent log
        _log("main_gui.py:__init__:2", "Skipping camera initialization (will do in background)", {}, "A")
        # #endregion
        
        # UI state
        self.current_detection = None
        self.stats = {'total_vehicles': 0, 'allowed_today': 0, 'denied_today': 0, 'total_detections': 0}
        
        # #region agent log
        _log("main_gui.py:__init__:7", "Before init_ui()", {}, "C")
        # #endregion
        self.init_ui()
        # #region agent log
        _log("main_gui.py:__init__:8", "After init_ui()", {}, "C")
        # #endregion
        
        # Video timer will be started after camera initializes
        self.video_timer = None
        if hasattr(self, 'video_label'):
            self.video_label.setText("⏳ Initializing Camera...")
            self.video_label.setStyleSheet("""
                background-color: #f9fafb;
                border: 2px dashed #d1d5db;
                border-radius: 12px;
                color: #f59e0b;
                font-size: 18px;
                font-weight: 500;
            """)
        
        # Update stats
        # #region agent log
        _log("main_gui.py:__init__:9", "Before update_stats()", {}, "E")
        # #endregion
        self.update_stats()
        # #region agent log
        _log("main_gui.py:__init__:10", "After update_stats()", {}, "E")
        # #endregion
        
        # Initialize camera and ALPR in background after GUI is shown
        QTimer.singleShot(100, self.init_camera_background)
        
        # #region agent log
        _log("main_gui.py:__init__:11", "ALPRMainWindow.__init__ completed", {}, "D")
        # #endregion
    
    def init_ui(self):
        """Initialize UI with Finance SaaS dashboard style"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background-color: #f9fafb;")
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)
        
        # Sidebar
        sidebar = self.create_sidebar()
        main_layout.addWidget(sidebar)
        
        # Main content area
        content_area = QWidget()
        content_area.setStyleSheet("background-color: #f9fafb;")
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_area.setLayout(content_layout)
        
        # Header
        header = self.create_header()
        content_layout.addWidget(header)
        
        # Stacked widget for pages
        self.stacked_widget = QStackedWidget()
        content_layout.addWidget(self.stacked_widget)
        
        # Create pages
        self.dashboard_page = self.create_dashboard_page()
        self.vehicles_page = self.create_vehicles_page()
        self.history_page = self.create_history_page()
        self.settings_page = self.create_settings_page()
        
        self.stacked_widget.addWidget(self.dashboard_page)
        self.stacked_widget.addWidget(self.vehicles_page)
        self.stacked_widget.addWidget(self.history_page)
        self.stacked_widget.addWidget(self.settings_page)
        
        main_layout.addWidget(content_area, 1)
    
    def create_sidebar(self):
        """Create modern sidebar navigation"""
        sidebar = QFrame()
        sidebar.setFrameShape(QFrame.StyledPanel)
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-right: 1px solid #e5e7eb;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 30, 20, 30)
        layout.setSpacing(0)
        sidebar.setLayout(layout)
        
        # Logo/Title
        logo = QLabel("🚗 ALPR System")
        logo.setStyleSheet("""
            color: #111827;
            font-size: 24px;
            font-weight: 700;
            padding-bottom: 30px;
        """)
        layout.addWidget(logo)
        
        # Navigation buttons
        nav_layout = QVBoxLayout()
        nav_layout.setSpacing(8)
        
        self.dashboard_btn = SidebarButton("📊 Dashboard")
        self.dashboard_btn.setChecked(True)
        self.dashboard_btn.clicked.connect(lambda: self.switch_page(0))
        nav_layout.addWidget(self.dashboard_btn)
        
        self.vehicles_btn = SidebarButton("🚗 Vehicles")
        self.vehicles_btn.clicked.connect(lambda: self.switch_page(1))
        nav_layout.addWidget(self.vehicles_btn)
        
        self.history_btn = SidebarButton("📜 History")
        self.history_btn.clicked.connect(lambda: self.switch_page(2))
        nav_layout.addWidget(self.history_btn)
        
        self.settings_btn = SidebarButton("⚙️ Settings")
        self.settings_btn.clicked.connect(lambda: self.switch_page(3))
        nav_layout.addWidget(self.settings_btn)
        
        layout.addLayout(nav_layout)
        layout.addStretch()
        
        # Footer
        footer = QLabel(f"Node: {self.config.get('node', {}).get('node_id', 'N/A')}")
        footer.setStyleSheet("color: #9ca3af; font-size: 12px; padding-top: 20px;")
        layout.addWidget(footer)
        
        return sidebar
    
    def create_header(self):
        """Create top header bar"""
        header = QFrame()
        header.setFrameShape(QFrame.StyledPanel)
        header.setFixedHeight(70)
        header.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-bottom: 1px solid #e5e7eb;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(30, 0, 30, 0)
        header.setLayout(layout)
        
        # Title (will be updated based on page)
        self.header_title = QLabel("Dashboard")
        self.header_title.setStyleSheet("color: #111827; font-size: 24px; font-weight: 700;")
        layout.addWidget(self.header_title)
        
        layout.addStretch()
        
        # Status indicator
        status = QLabel("● Active")
        status.setStyleSheet("color: #10b981; font-size: 14px; font-weight: 500;")
        layout.addWidget(status)
        
        return header
    
    def switch_page(self, index):
        """Switch between pages"""
        # Update button states
        self.dashboard_btn.setChecked(index == 0)
        self.vehicles_btn.setChecked(index == 1)
        self.history_btn.setChecked(index == 2)
        self.settings_btn.setChecked(index == 3)
        
        # Update header title
        titles = ["Dashboard", "Vehicles", "History", "Settings"]
        self.header_title.setText(titles[index])
        
        # Switch page
        self.stacked_widget.setCurrentIndex(index)
        
        # Refresh data if needed
        if index == 1:
            self.refresh_vehicles()
        elif index == 2:
            self.refresh_history()
        elif index == 0:
            self.update_stats()
    
    def create_dashboard_page(self):
        """Create dashboard page with metrics and video"""
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(24)
        page.setLayout(layout)
        
        # Metrics row
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(20)
        
        self.total_vehicles_card = MetricCard("Total Vehicles", "0", "Registered in system")
        metrics_layout.addWidget(self.total_vehicles_card)
        
        self.allowed_card = MetricCard("Allowed Today", "0", "Authorized entries")
        metrics_layout.addWidget(self.allowed_card)
        
        self.denied_card = MetricCard("Denied Today", "0", "Unauthorized attempts")
        metrics_layout.addWidget(self.denied_card)
        
        self.detections_card = MetricCard("Total Detections", "0", "All time")
        metrics_layout.addWidget(self.detections_card)
        
        layout.addLayout(metrics_layout)
        
        # Video and status row
        video_row = QHBoxLayout()
        video_row.setSpacing(20)
        
        # Video feed card
        video_card = QFrame()
        video_card.setFrameShape(QFrame.StyledPanel)
        video_card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }
        """)
        video_layout = QVBoxLayout()
        video_layout.setContentsMargins(20, 20, 20, 20)
        video_card.setLayout(video_layout)
        
        video_title = QLabel("Live Camera Feed")
        video_title.setStyleSheet("color: #111827; font-size: 18px; font-weight: 600; padding-bottom: 15px;")
        video_layout.addWidget(video_title)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setMaximumSize(640, 360)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border-radius: 8px;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        
        # Preprocessed plate display
        preprocess_title = QLabel("Preprocessed Plate (Sent to OCR)")
        preprocess_title.setStyleSheet("color: #111827; font-size: 14px; font-weight: 600; padding-top: 15px; padding-bottom: 5px;")
        video_layout.addWidget(preprocess_title)
        
        self.preprocessed_label = QLabel()
        self.preprocessed_label.setMinimumSize(320, 80)
        self.preprocessed_label.setMaximumSize(640, 120)
        self.preprocessed_label.setStyleSheet("""
            QLabel {
                background-color: #1f2937;
                border-radius: 8px;
                border: 2px solid #374151;
            }
        """)
        self.preprocessed_label.setAlignment(Qt.AlignCenter)
        self.preprocessed_label.setText("No plate detected")
        self.preprocessed_label.setStyleSheet("""
            QLabel {
                background-color: #1f2937;
                border-radius: 8px;
                border: 2px solid #374151;
                color: #9ca3af;
                font-size: 12px;
            }
        """)
        video_layout.addWidget(self.preprocessed_label)
        
        # Buttons layout (horizontal)
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        # Resume button
        self.resume_btn = QPushButton("▶️  Resume Detection")
        self.resume_btn.clicked.connect(self.resume_detection)
        self.resume_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
            QPushButton:disabled {
                background-color: #d1d5db;
                color: #9ca3af;
            }
        """)
        self.resume_btn.setEnabled(False)  # Disabled initially
        buttons_layout.addWidget(self.resume_btn)
        
        # Upload Image button
        self.upload_btn = QPushButton("📤  Upload Image")
        self.upload_btn.clicked.connect(self.upload_and_process_image)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
            QPushButton:disabled {
                background-color: #d1d5db;
                color: #9ca3af;
            }
        """)
        buttons_layout.addWidget(self.upload_btn)
        
        video_layout.addLayout(buttons_layout)
        
        # Status cards
        status_layout = QVBoxLayout()
        status_layout.setSpacing(15)
        
        # Plate display
        plate_card = QFrame()
        plate_card.setFrameShape(QFrame.StyledPanel)
        plate_card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }
        """)
        plate_layout = QVBoxLayout()
        plate_layout.setContentsMargins(20, 20, 20, 20)
        plate_card.setLayout(plate_layout)
        
        plate_title = QLabel("Detected Plate")
        plate_title.setStyleSheet("color: #6b7280; font-size: 14px; font-weight: 500;")
        plate_layout.addWidget(plate_title)
        
        self.plate_label = QLabel("NO DETECTION")
        self.plate_label.setStyleSheet("color: #111827; font-size: 32px; font-weight: 700;")
        plate_layout.addWidget(self.plate_label)
        status_layout.addWidget(plate_card)
        
        # Status display
        status_card = QFrame()
        status_card.setFrameShape(QFrame.StyledPanel)
        status_card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }
        """)
        status_card_layout = QVBoxLayout()
        status_card_layout.setContentsMargins(20, 20, 20, 20)
        status_card.setLayout(status_card_layout)
        
        status_title = QLabel("Status")
        status_title.setStyleSheet("color: #6b7280; font-size: 14px; font-weight: 500;")
        status_card_layout.addWidget(status_title)
        
        self.status_label = QLabel("WAITING")
        self.status_label.setStyleSheet("color: #6b7280; font-size: 24px; font-weight: 600;")
        status_card_layout.addWidget(self.status_label)
        status_layout.addWidget(status_card)
        
        # Vehicle info
        info_card = QFrame()
        info_card.setFrameShape(QFrame.StyledPanel)
        info_card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }
        """)
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(20, 20, 20, 20)
        info_card.setLayout(info_layout)
        
        info_title = QLabel("Vehicle Information")
        info_title.setStyleSheet("color: #6b7280; font-size: 14px; font-weight: 500; padding-bottom: 10px;")
        info_layout.addWidget(info_title)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(120)
        self.info_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 10px;
                color: #111827;
                font-size: 13px;
            }
        """)
        info_layout.addWidget(self.info_text)
        status_layout.addWidget(info_card)
        
        video_row.addWidget(video_card)
        video_row.addLayout(status_layout)
        
        layout.addLayout(video_row)
        layout.addStretch()
        
        return page
    
    def create_vehicles_page(self):
        """Create vehicles management page"""
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(24)
        page.setLayout(layout)
        
        # Add vehicle form
        form_card = QFrame()
        form_card.setFrameShape(QFrame.StyledPanel)
        form_card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }
        """)
        form_layout = QVBoxLayout()
        form_layout.setContentsMargins(24, 24, 24, 24)
        form_card.setLayout(form_layout)
        
        form_title = QLabel("Add New Vehicle")
        form_title.setStyleSheet("color: #111827; font-size: 20px; font-weight: 600; padding-bottom: 20px;")
        form_layout.addWidget(form_title)
        
        # Form fields in grid
        fields_layout = QHBoxLayout()
        fields_layout.setSpacing(15)
        
        # Plate
        plate_group = QVBoxLayout()
        plate_label = QLabel("License Plate")
        plate_label.setStyleSheet("color: #374151; font-size: 14px; font-weight: 500; padding-bottom: 8px;")
        plate_group.addWidget(plate_label)
        self.plate_input = QLineEdit()
        self.plate_input.setPlaceholderText("WB-01-AB-1234")
        self.plate_input.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #3b82f6;
            }
        """)
        plate_group.addWidget(self.plate_input)
        fields_layout.addLayout(plate_group)
        
        # Owner
        owner_group = QVBoxLayout()
        owner_label = QLabel("Owner Name")
        owner_label.setStyleSheet("color: #374151; font-size: 14px; font-weight: 500; padding-bottom: 8px;")
        owner_group.addWidget(owner_label)
        self.owner_input = QLineEdit()
        self.owner_input.setPlaceholderText("Full name")
        self.owner_input.setStyleSheet(self.plate_input.styleSheet())
        owner_group.addWidget(self.owner_input)
        fields_layout.addLayout(owner_group)
        
        # Type
        type_group = QVBoxLayout()
        type_label = QLabel("Vehicle Type")
        type_label.setStyleSheet("color: #374151; font-size: 14px; font-weight: 500; padding-bottom: 8px;")
        type_group.addWidget(type_label)
        self.type_input = QComboBox()
        self.type_input.addItems(['Car', 'SUV', 'Truck', 'Bus', 'Motorcycle'])
        self.type_input.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }
            QComboBox:focus {
                border: 2px solid #3b82f6;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                selection-background-color: #3b82f6;
            }
        """)
        type_group.addWidget(self.type_input)
        fields_layout.addLayout(type_group)
        
        # Contact
        contact_group = QVBoxLayout()
        contact_label = QLabel("Contact")
        contact_label.setStyleSheet("color: #374151; font-size: 14px; font-weight: 500; padding-bottom: 8px;")
        contact_group.addWidget(contact_label)
        self.contact_input = QLineEdit()
        self.contact_input.setPlaceholderText("Phone number")
        self.contact_input.setStyleSheet(self.plate_input.styleSheet())
        contact_group.addWidget(self.contact_input)
        fields_layout.addLayout(contact_group)
        
        form_layout.addLayout(fields_layout)
        
        # Add button
        add_btn = QPushButton("Add Vehicle")
        add_btn.clicked.connect(self.add_vehicle)
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """)
        add_btn.setMaximumWidth(200)
        form_layout.addWidget(add_btn, alignment=Qt.AlignLeft)
        
        layout.addWidget(form_card)
        
        # Vehicles table
        table_card = QFrame()
        table_card.setFrameShape(QFrame.StyledPanel)
        table_card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }
        """)
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(24, 24, 24, 24)
        table_card.setLayout(table_layout)
        
        table_title = QLabel("Registered Vehicles")
        table_title.setStyleSheet("color: #111827; font-size: 20px; font-weight: 600; padding-bottom: 20px;")
        table_layout.addWidget(table_title)
        
        self.vehicle_table = QTableWidget()
        self.vehicle_table.setColumnCount(5)
        self.vehicle_table.setHorizontalHeaderLabels(['Plate', 'Owner', 'Type', 'Contact', 'Actions'])
        self.vehicle_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.vehicle_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                gridline-color: #e5e7eb;
            }
            QTableWidget::item {
                padding: 12px;
                border-bottom: 1px solid #f3f4f6;
            }
            QTableWidget::item:selected {
                background-color: #eff6ff;
            }
            QHeaderView::section {
                background-color: #f9fafb;
                color: #374151;
                padding: 12px;
                border: none;
                border-bottom: 2px solid #e5e7eb;
                font-weight: 600;
            }
        """)
        table_layout.addWidget(self.vehicle_table)
        
        layout.addWidget(table_card)
        
        return page
    
    def create_history_page(self):
        """Create history page"""
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(24)
        page.setLayout(layout)
        
        # History table card
        table_card = QFrame()
        table_card.setFrameShape(QFrame.StyledPanel)
        table_card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }
        """)
        table_layout = QVBoxLayout()
        table_layout.setContentsMargins(24, 24, 24, 24)
        table_card.setLayout(table_layout)
        
        title_layout = QHBoxLayout()
        title = QLabel("Detection History")
        title.setStyleSheet("color: #111827; font-size: 20px; font-weight: 600;")
        title_layout.addWidget(title)
        title_layout.addStretch()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_history)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #f3f4f6;
                color: #374151;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #e5e7eb;
            }
        """)
        title_layout.addWidget(refresh_btn)
        table_layout.addLayout(title_layout)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(['Time', 'Node', 'Plate', 'Status', 'Owner'])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                gridline-color: #e5e7eb;
            }
            QTableWidget::item {
                padding: 12px;
                border-bottom: 1px solid #f3f4f6;
            }
            QTableWidget::item:selected {
                background-color: #eff6ff;
            }
            QHeaderView::section {
                background-color: #f9fafb;
                color: #374151;
                padding: 12px;
                border: none;
                border-bottom: 2px solid #e5e7eb;
                font-weight: 600;
            }
        """)
        table_layout.addWidget(self.history_table)
        
        layout.addWidget(table_card)
        
        return page
    
    def create_settings_page(self):
        """Create settings page"""
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(24)
        page.setLayout(layout)
        
        # Settings card
        settings_card = QFrame()
        settings_card.setFrameShape(QFrame.StyledPanel)
        settings_card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }
        """)
        settings_layout = QVBoxLayout()
        settings_layout.setContentsMargins(24, 24, 24, 24)
        settings_card.setLayout(settings_layout)
        
        title = QLabel("System Configuration")
        title.setStyleSheet("color: #111827; font-size: 20px; font-weight: 600; padding-bottom: 20px;")
        settings_layout.addWidget(title)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        # #region agent log
        _log("main_gui.py:240", "Accessing config keys for settings", {"has_node": "node" in self.config, "has_database": "database" in self.config, "has_camera": "camera" in self.config, "has_yolo": "yolo" in self.config}, "B")
        # #endregion
        try:
            node_id = self.config['node']['node_id']
            location = self.config['node']['location']
            db_host = self.config['database']['host']
            db_port = self.config['database']['port']
            camera_source = self.config['camera']['source']
            model_path = self.config['yolo']['model_path']
            device = self.config['yolo']['device']
            camera_status = "✅ Connected" if (self.camera and self.camera.camera_available) else "❌ Not Available"
        except KeyError as e:
            # #region agent log
            _log("main_gui.py:240", "Missing nested config key", {"error": str(e)}, "B")
            # #endregion
            node_id = "N/A"
            location = "N/A"
            db_host = "N/A"
            db_port = "N/A"
            camera_source = "N/A"
            model_path = "N/A"
            device = "N/A"
            camera_status = "❌ Unknown"
        
        info_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 15px;
                color: #111827;
                font-size: 14px;
            }
        """)
        info_text.setHtml(f"""
            <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;'>
                <table style='width: 100%; border-collapse: collapse;'>
                    <tr style='border-bottom: 1px solid #e5e7eb;'>
                        <td style='padding: 12px 0; color: #6b7280; font-weight: 500; width: 200px;'>Node ID</td>
                        <td style='padding: 12px 0; color: #111827; font-weight: 500;'>{node_id}</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #e5e7eb;'>
                        <td style='padding: 12px 0; color: #6b7280; font-weight: 500;'>Location</td>
                        <td style='padding: 12px 0; color: #111827; font-weight: 500;'>{location}</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #e5e7eb;'>
                        <td style='padding: 12px 0; color: #6b7280; font-weight: 500;'>Database</td>
                        <td style='padding: 12px 0; color: #111827; font-weight: 500;'>{db_host}:{db_port}</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #e5e7eb;'>
                        <td style='padding: 12px 0; color: #6b7280; font-weight: 500;'>Camera</td>
                        <td style='padding: 12px 0; color: #111827; font-weight: 500;'>{camera_source} - {camera_status}</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #e5e7eb;'>
                        <td style='padding: 12px 0; color: #6b7280; font-weight: 500;'>YOLO Model</td>
                        <td style='padding: 12px 0; color: #111827; font-weight: 500;'>{model_path}</td>
                    </tr>
                    <tr>
                        <td style='padding: 12px 0; color: #6b7280; font-weight: 500;'>Device</td>
                        <td style='padding: 12px 0; color: #111827; font-weight: 500;'>{device.upper()}</td>
                    </tr>
                </table>
            </div>
        """)
        settings_layout.addWidget(info_text)
        
        layout.addWidget(settings_card)
        layout.addStretch()
        
        return page
    
    def update_stats(self):
        """Update dashboard statistics"""
        # #region agent log
        _log("main_gui.py:update_stats:1", "update_stats() started", {}, "E")
        # #endregion
        try:
            # #region agent log
            _log("main_gui.py:update_stats:2", "Before db.get_all_vehicles()", {}, "E")
            # #endregion
            vehicles = self.db.get_all_vehicles()
            # #region agent log
            _log("main_gui.py:update_stats:3", "After db.get_all_vehicles(), before db.get_detection_history()", {"vehicle_count": len(vehicles)}, "E")
            # #endregion
            history = self.db.get_detection_history(1000)
            
            today = datetime.now().date()
            allowed_today = sum(1 for h in history if h['status'] == 'ALLOWED' and h['detected_at'].date() == today)
            denied_today = sum(1 for h in history if h['status'] == 'DENIED' and h['detected_at'].date() == today)
            
            self.stats['total_vehicles'] = len(vehicles)
            self.stats['allowed_today'] = allowed_today
            self.stats['denied_today'] = denied_today
            self.stats['total_detections'] = len(history)
            
            # Update cards
            if hasattr(self, 'total_vehicles_card'):
                self.total_vehicles_card.set_value(self.stats['total_vehicles'])
            if hasattr(self, 'allowed_card'):
                self.allowed_card.set_value(self.stats['allowed_today'])
            if hasattr(self, 'denied_card'):
                self.denied_card.set_value(self.stats['denied_today'])
            if hasattr(self, 'detections_card'):
                self.detections_card.set_value(self.stats['total_detections'])
            # #region agent log
            _log("main_gui.py:update_stats:4", "update_stats() completed successfully", {}, "E")
            # #endregion
        except Exception as e:
            # #region agent log
            _log("main_gui.py:update_stats:5", "update_stats() exception", {"error": str(e)}, "E")
            # #endregion
            pass
    
    def update_video(self):
        """Update video feed"""
        if not self.camera or not self.camera.camera_available:
            return
        
        # #region agent log
        _log("main_gui.py:255", "Reading camera frame", {"camera_exists": self.camera is not None}, "D")
        # #endregion
        ret, frame = self.camera.read()
        # #region agent log
        _log("main_gui.py:256", "Camera read result", {"ret": ret, "frame_is_none": frame is None, "frame_shape": frame.shape if frame is not None else None}, "D")
        # #endregion
        if ret and frame is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
            except Exception as e:
                # #region agent log
                _log("main_gui.py:268", "Error updating video", {"error": str(e)}, "D")
                # #endregion
                pass
    
    def handle_detection(self, result):
        """Handle detection result from thread"""
        print("\n" + "🔔 "*20)
        print("🔔 HANDLE_DETECTION CALLED!")
        print("🔔 "*20)
        
        # #region agent log
        _log("main_gui.py:handle_detection:1", "handle_detection() called", {"result_keys": list(result.keys()), "plate_number": result.get('plate_number'), "confidence": result.get('confidence')}, "O")
        # #endregion
        
        try:
            plate_number = result['plate_number']
            confidence = result['confidence']
            frame = result.get('frame')
            
            print(f"📋 Plate Number: {plate_number}")
            print(f"📊 Confidence: {confidence:.2%}")
            
            # Stop video timer to freeze the display
            if self.video_timer and self.video_timer.isActive():
                self.video_timer.stop()
                print("⏸️  Video timer stopped")
            
            # Save and display snapshot
            if frame is not None:
                # Create snapshots directory if it doesn't exist
                os.makedirs('snapshots', exist_ok=True)
                
                # Save snapshot with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                snapshot_path = f'snapshots/{plate_number}_{timestamp}.jpg'
                cv2.imwrite(snapshot_path, frame)
                print(f"📸 Snapshot saved: {snapshot_path}")
                
                # Display the snapshot
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.video_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
                print(f"🖼️  Snapshot displayed on screen")
            
            # Display preprocessed plate image
            preprocessed_img = result.get('preprocessed_image')
            if preprocessed_img is not None:
                # Convert grayscale to RGB for display
                if len(preprocessed_img.shape) == 2:
                    preprocessed_rgb = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2RGB)
                else:
                    preprocessed_rgb = preprocessed_img
                
                h, w, ch = preprocessed_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(preprocessed_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.preprocessed_label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.preprocessed_label.setPixmap(scaled_pixmap)
                print(f"🔍 Preprocessed plate image displayed")
            
            # #region agent log
            _log("main_gui.py:handle_detection:2", "Before updating plate_label", {"plate_number": plate_number}, "O")
            # #endregion
            
            print(f"🖼️  Updating plate label to: {plate_number}")
            self.plate_label.setText(plate_number)
            print(f"✅ Plate label updated")
            
            # #region agent log
            _log("main_gui.py:handle_detection:3", "Before get_vehicle()", {"plate_number": plate_number}, "O")
            # #endregion
            
            print(f"🔍 Checking database for vehicle: {plate_number}")
            vehicle = self.db.get_vehicle(plate_number)
            print(f"📦 Database result: {'FOUND' if vehicle else 'NOT FOUND'}")
            
            # #region agent log
            _log("main_gui.py:handle_detection:4", "After get_vehicle()", {"vehicle_found": vehicle is not None}, "O")
            # #endregion
            
            if vehicle:
                print(f"\n✅ VEHICLE FOUND IN DATABASE!")
                print(f"   Owner: {vehicle.get('owner_name', 'N/A')}")
                print(f"   Type: {vehicle.get('vehicle_type', 'N/A')}")
                
                print(f"🟢 Setting status to ALLOWED")
                self.status_label.setText("✓ ALLOWED")
                self.status_label.setStyleSheet("color: #10b981; font-size: 24px; font-weight: 600;")
                print(f"✅ Status label set to ALLOWED")
                
                info_html = f"""
                    <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;'>
                        <p style='color: #111827; margin: 4px 0;'><b>Owner:</b> {vehicle['owner_name']}</p>
                        <p style='color: #111827; margin: 4px 0;'><b>Type:</b> {vehicle['vehicle_type']}</p>
                        <p style='color: #111827; margin: 4px 0;'><b>Contact:</b> {vehicle['contact_number']}</p>
                        <p style='color: #6b7280; margin: 4px 0;'><b>Confidence:</b> {confidence:.1%}</p>
                    </div>
                """
                self.info_text.setHtml(info_html)
                print(f"✅ Vehicle info displayed")
                
                try:
                    print(f"💾 Logging ALLOWED detection to database...")
                    self.db.log_detection(
                        self.config['node']['node_id'],
                        plate_number,
                        confidence,
                        'ALLOWED',
                        vehicle['owner_name']
                    )
                    print(f"✅ Detection logged successfully")
                except Exception as e:
                    # #region agent log
                    _log("main_gui.py:handle_detection:5", "Error logging ALLOWED detection", {"error": str(e)}, "O")
                    # #endregion
                    print(f"❌ Error logging detection: {e}")
            else:
                print(f"\n⛔ VEHICLE NOT FOUND IN DATABASE!")
                
                print(f"🔴 Setting status to DENIED")
                self.status_label.setText("✗ DENIED")
                self.status_label.setStyleSheet("color: #ef4444; font-size: 24px; font-weight: 600;")
                print(f"✅ Status label set to DENIED")
                
                info_html = f"""
                    <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;'>
                        <p style='color: #111827; margin: 4px 0;'><b>Plate:</b> {plate_number}</p>
                        <p style='color: #ef4444; margin: 4px 0;'><b>Status:</b> Not registered</p>
                        <p style='color: #6b7280; margin: 4px 0;'><b>Confidence:</b> {confidence:.1%}</p>
                    </div>
                """
                self.info_text.setHtml(info_html)
                print(f"✅ Denial info displayed")
                
                try:
                    print(f"💾 Logging DENIED detection to database...")
                    self.db.log_detection(
                        self.config['node']['node_id'],
                        plate_number,
                        confidence,
                        'DENIED'
                    )
                    print(f"✅ Detection logged successfully")
                except Exception as e:
                    # #region agent log
                    _log("main_gui.py:handle_detection:6", "Error logging DENIED detection", {"error": str(e)}, "O")
                    # #endregion
                    print(f"❌ Error logging detection: {e}")
            
            # Refresh UI (wrapped in try-except to not block status display)
            try:
                print(f"🔄 Refreshing UI (history and stats)...")
                self.refresh_history()
                self.update_stats()
                print(f"✅ UI refreshed")
            except Exception as e:
                # #region agent log
                _log("main_gui.py:handle_detection:7", "Error refreshing UI", {"error": str(e)}, "O")
                # #endregion
                print(f"❌ Error refreshing UI: {e}")
            
            # Enable resume button
            if hasattr(self, 'resume_btn'):
                self.resume_btn.setEnabled(True)
                print(f"✅ Resume button enabled")
            
            print("\n" + "="*60)
            print("🎉 HANDLE_DETECTION COMPLETED SUCCESSFULLY!")
            print("🎉 Detection paused - Click 'Resume Detection' to continue")
            print("="*60 + "\n")
            
        except Exception as e:
            # #region agent log
            _log("main_gui.py:handle_detection:8", "Critical error in handle_detection", {"error": str(e), "error_type": type(e).__name__}, "O")
            # #endregion
            print("\n" + "❌"*20)
            print(f"❌ CRITICAL ERROR IN HANDLE_DETECTION!")
            print(f"❌ Error: {e}")
            print(f"❌ Type: {type(e).__name__}")
            print("❌"*20 + "\n")
            
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            
            # Ensure status is updated even if there's an error
            self.status_label.setText("⚠ ERROR")
            self.status_label.setStyleSheet("color: #ef4444; font-size: 24px; font-weight: 600;")
            
            info_html = f"""
                <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;'>
                    <p style='color: #ef4444; margin: 4px 0;'><b>Error:</b> {str(e)}</p>
                    <p style='color: #6b7280; margin: 4px 0; font-size: 12px;'>Check console for details</p>
                </div>
            """
            self.info_text.setHtml(info_html)
            
            # Enable resume button even on error
            if hasattr(self, 'resume_btn'):
                self.resume_btn.setEnabled(True)
    
    def reset_display(self):
        """Reset display to waiting state"""
        print("\n" + "⏱️ "*20)
        print("⏱️  RESETTING DISPLAY TO WAITING STATE")
        print("⏱️ "*20 + "\n")
        
        self.plate_label.setText("NO DETECTION")
        self.status_label.setText("WAITING")
        self.status_label.setStyleSheet("color: #6b7280; font-size: 24px; font-weight: 600;")
        self.info_text.clear()
        
        # Clear preprocessed plate display
        if hasattr(self, 'preprocessed_label'):
            self.preprocessed_label.clear()
            self.preprocessed_label.setText("No plate detected")
            self.preprocessed_label.setStyleSheet("""
                QLabel {
                    background-color: #1f2937;
                    border-radius: 8px;
                    border: 2px solid #374151;
                    color: #9ca3af;
                    font-size: 12px;
                }
            """)
        
        print("✅ Display reset complete\n")
    
    def resume_detection(self):
        """Resume detection after a plate was detected"""
        print("\n" + "▶️ "*20)
        print("▶️  RESUMING DETECTION")
        print("▶️ "*20 + "\n")
        
        # Reset display
        self.reset_display()
        
        # Resume video timer
        if self.video_timer and self.camera and self.camera.camera_available:
            if not self.video_timer.isActive():
                self.video_timer.start(30)
                print("▶️  Video timer resumed")
        
        # Resume detection thread
        if self.detection_thread:
            self.detection_thread.resume()
            print("▶️  Detection thread resumed")
        
        # Disable resume button
        if hasattr(self, 'resume_btn'):
            self.resume_btn.setEnabled(False)
            print("✅ Resume button disabled")
        
        print("\n" + "="*60)
        print("🎉 DETECTION RESUMED - READY FOR NEXT VEHICLE!")
        print("="*60 + "\n")
    
    def upload_and_process_image(self):
        """Upload an image file and process it for license plate detection"""
        print("\n" + "📤 "*20)
        print("📤  UPLOAD IMAGE FOR PROCESSING")
        print("📤 "*20 + "\n")
        
        # Open file dialog to select image
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select License Plate Image",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*.*)"
        )
        
        if not file_path:
            print("❌ No file selected")
            return
        
        print(f"📁 Selected file: {file_path}")
        
        try:
            # Load image using OpenCV
            frame = cv2.imread(file_path)
            
            if frame is None:
                QMessageBox.warning(self, "Error", "Failed to load image. Please select a valid image file.")
                print(f"❌ Failed to load image: {file_path}")
                return
            
            print(f"✅ Image loaded successfully: {frame.shape}")
            
            # Check if ALPR engine is initialized
            if not hasattr(self, 'alpr_engine') or self.alpr_engine is None:
                QMessageBox.warning(self, "Error", "ALPR engine not initialized. Please wait for the system to fully start.")
                print("❌ ALPR engine not initialized")
                return
            
            print("🔍 Processing image through ALPR engine...")
            
            # Process image through ALPR engine
            result = self.alpr_engine.process_frame(frame)
            
            if result is None:
                QMessageBox.information(self, "No Detection", "No license plate detected in the uploaded image.")
                print("❌ No plate detected in uploaded image")
                return
            
            print(f"✅ Plate detected: {result.get('plate_number')} (Confidence: {result.get('confidence'):.2%})")
            
            # Add frame to result for display
            result['frame'] = frame
            
            # Stop video timer if running
            if hasattr(self, 'video_timer') and self.video_timer and self.video_timer.isActive():
                self.video_timer.stop()
                print("⏸️  Video timer stopped")
            
            # Pause detection thread if running
            if hasattr(self, 'detection_thread') and self.detection_thread:
                self.detection_thread.pause()
                print("⏸️  Detection thread paused")
            
            # Process and display the detection result
            self.handle_detection(result)
            
            print("\n" + "="*60)
            print("🎉 IMAGE PROCESSING COMPLETED!")
            print("="*60 + "\n")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process image:\n{str(e)}")
            print(f"❌ Error processing image: {e}")
            import traceback
            traceback.print_exc()
    
    def add_vehicle(self):
        """Add new vehicle to database"""
        plate = self.plate_input.text().strip().upper()
        owner = self.owner_input.text().strip()
        vehicle_type = self.type_input.currentText()
        contact = self.contact_input.text().strip()
        
        if not plate or not owner:
            QMessageBox.warning(self, "Error", "Plate number and owner name are required!")
            return
        
        success = self.db.add_vehicle(
            plate_number=plate,
            owner_name=owner,
            vehicle_type=vehicle_type,
            contact_number=contact
        )
        
        if success:
            QMessageBox.information(self, "Success", "Vehicle added successfully!")
            self.plate_input.clear()
            self.owner_input.clear()
            self.contact_input.clear()
            self.refresh_vehicles()
            self.update_stats()
        else:
            QMessageBox.warning(self, "Error", "Failed to add vehicle. Plate may already exist.")
    
    def refresh_vehicles(self):
        """Refresh vehicle table"""
        vehicles = self.db.get_all_vehicles()
        self.vehicle_table.setRowCount(len(vehicles))
        
        for i, vehicle in enumerate(vehicles):
            self.vehicle_table.setItem(i, 0, QTableWidgetItem(vehicle['plate_number']))
            self.vehicle_table.setItem(i, 1, QTableWidgetItem(vehicle['owner_name']))
            self.vehicle_table.setItem(i, 2, QTableWidgetItem(vehicle['vehicle_type'] or ''))
            self.vehicle_table.setItem(i, 3, QTableWidgetItem(vehicle['contact_number'] or ''))
            
            delete_btn = QPushButton("Delete")
            delete_btn.clicked.connect(lambda checked, p=vehicle['plate_number']: self.delete_vehicle(p))
            delete_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ef4444;
                    color: #ffffff;
                    border: none;
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-size: 12px;
                    font-weight: 500;
                }
                QPushButton:hover {
                    background-color: #dc2626;
                }
            """)
            self.vehicle_table.setCellWidget(i, 4, delete_btn)
    
    def delete_vehicle(self, plate_number):
        """Delete vehicle from database"""
        reply = QMessageBox.question(
            self, 'Confirm Delete',
            f'Are you sure you want to delete vehicle {plate_number}?',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.db.delete_vehicle(plate_number)
            self.refresh_vehicles()
            self.update_stats()
    
    def refresh_history(self):
        """Refresh detection history"""
        history = self.db.get_detection_history(50)
        self.history_table.setRowCount(len(history))
        
        for i, record in enumerate(history):
            timestamp = record['detected_at'].strftime('%Y-%m-%d %H:%M:%S')
            self.history_table.setItem(i, 0, QTableWidgetItem(timestamp))
            self.history_table.setItem(i, 1, QTableWidgetItem(record['node_id']))
            self.history_table.setItem(i, 2, QTableWidgetItem(record['plate_number']))
            
            status_item = QTableWidgetItem(record['status'])
            if record['status'] == 'ALLOWED':
                status_item.setForeground(QColor(16, 185, 129))
                status_item.setBackground(QColor(209, 250, 229, 50))
            else:
                status_item.setForeground(QColor(239, 68, 68))
                status_item.setBackground(QColor(254, 226, 226, 50))
            status_item.setFont(QFont('Segoe UI', 10, QFont.Bold))
            self.history_table.setItem(i, 3, status_item)
            
            self.history_table.setItem(i, 4, QTableWidgetItem(record['owner_name'] or 'Unknown'))
    
    def init_camera_background(self):
        """Initialize camera in background thread"""
        if self.camera_loading:
            return
        
        self.camera_loading = True
        
        # Show loading status
        if hasattr(self, 'status_label'):
            self.status_label.setText("LOADING CAMERA...")
            self.status_label.setStyleSheet("color: #f59e0b; font-size: 24px; font-weight: 600;")
        
        # Start initialization thread
        self.camera_init_thread = CameraInitThread()
        self.camera_init_thread.initialization_status.connect(self.on_camera_status)
        self.camera_init_thread.initialization_complete.connect(self.on_camera_initialized)
        self.camera_init_thread.start()
        
        # Set a timeout of 10 seconds for camera initialization
        QTimer.singleShot(10000, self.on_camera_init_timeout)
    
    def on_camera_status(self, message):
        """Handle camera initialization status updates"""
        print(message)
        if hasattr(self, 'video_label'):
            self.video_label.setText(f"⏳ {message}")
    
    def on_camera_init_timeout(self):
        """Handle camera initialization timeout"""
        if not self.camera_loading:
            return  # Already completed
        
        # #region agent log
        _log("main_gui.py:on_camera_init_timeout", "Camera initialization timed out after 10 seconds", {"thread_is_running": self.camera_init_thread.isRunning() if hasattr(self, 'camera_init_thread') else False}, "J")
        # #endregion
        
        print("⚠ Camera initialization timed out (10 seconds)")
        self.camera_loading = False
        
        # Terminate the hung thread (forcefully)
        if hasattr(self, 'camera_init_thread') and self.camera_init_thread.isRunning():
            self.camera_init_thread.terminate()
            self.camera_init_thread.wait(1000)
        
        # Show timeout message
        if hasattr(self, 'video_label'):
            self.video_label.setText("⚠ Camera Timeout\n(Check camera permissions/availability)")
            self.video_label.setStyleSheet("""
                background-color: #f9fafb;
                border: 2px dashed #d1d5db;
                border-radius: 12px;
                color: #ef4444;
                font-size: 16px;
                font-weight: 500;
            """)
        
        # Reset status
        if hasattr(self, 'status_label'):
            self.status_label.setText("WAITING")
            self.status_label.setStyleSheet("color: #6b7280; font-size: 24px; font-weight: 600;")
    
    def on_camera_initialized(self, camera):
        """Handle camera initialization completion"""
        if not self.camera_loading:
            return  # Timeout already occurred
        
        self.camera = camera
        self.camera_loading = False
        
        if self.camera is not None and self.camera.camera_available:
            # Camera initialized successfully - start video timer
            self.video_timer = QTimer()
            self.video_timer.timeout.connect(self.update_video)
            self.video_timer.start(30)
            
            # Update video label
            if hasattr(self, 'video_label'):
                self.video_label.setText("")
            
            # Now initialize ALPR engine
            QTimer.singleShot(100, self.init_alpr_background)
        else:
            # Camera failed to initialize
            if hasattr(self, 'video_label'):
                self.video_label.setText("⚠ Camera Not Available")
                self.video_label.setStyleSheet("""
                    background-color: #f9fafb;
                    border: 2px dashed #d1d5db;
                    border-radius: 12px;
                    color: #6b7280;
                    font-size: 18px;
                    font-weight: 500;
                """)
            
            # Reset status
            if hasattr(self, 'status_label'):
                self.status_label.setText("WAITING")
                self.status_label.setStyleSheet("color: #6b7280; font-size: 24px; font-weight: 600;")
    
    def init_alpr_background(self):
        """Initialize ALPR engine in background thread"""
        if self.alpr_loading:
            return
        
        self.alpr_loading = True
        
        # Show loading status
        if hasattr(self, 'status_label'):
            self.status_label.setText("LOADING ALPR...")
            self.status_label.setStyleSheet("color: #f59e0b; font-size: 24px; font-weight: 600;")
        
        # Start initialization thread
        self.alpr_init_thread = ALPRInitThread()
        self.alpr_init_thread.initialization_status.connect(self.on_alpr_status)
        self.alpr_init_thread.initialization_complete.connect(self.on_alpr_initialized)
        self.alpr_init_thread.start()
    
    def on_alpr_status(self, message):
        """Handle ALPR initialization status updates"""
        print(message)
        if hasattr(self, 'info_text'):
            self.info_text.setHtml(f"""
                <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;'>
                    <p style='color: #f59e0b; margin: 4px 0;'><b>Status:</b> {message}</p>
                    <p style='color: #6b7280; margin: 4px 0; font-size: 12px;'>This may take a moment...</p>
                </div>
            """)
    
    def on_alpr_initialized(self, alpr_engine):
        """Handle ALPR initialization completion"""
        print("\n" + "="*60)
        print("🤖 ON_ALPR_INITIALIZED CALLED")
        print("="*60)
        
        self.alpr_engine = alpr_engine
        self.alpr_loading = False
        
        if self.alpr_engine is not None:
            print("✅ ALPR Engine initialized successfully")
            
            # Start detection thread
            if self.camera and self.camera.camera_available:
                print("📷 Camera available, starting detection thread...")
                self.detection_thread = DetectionThread(self.camera, self.alpr_engine)
                print("🔗 Connecting detection_result signal to handle_detection...")
                self.detection_thread.detection_result.connect(self.handle_detection)
                print("✅ Signal connected!")
                print("▶️  Starting detection thread...")
                self.detection_thread.start()
                print("✅ Detection thread started!")
                
                # Reset status display
                if hasattr(self, 'status_label'):
                    self.status_label.setText("WAITING")
                    self.status_label.setStyleSheet("color: #6b7280; font-size: 24px; font-weight: 600;")
                    print("✅ Status label set to WAITING")
                
                # Clear info text
                if hasattr(self, 'info_text'):
                    self.info_text.clear()
                    print("✅ Info text cleared")
                
                print("\n🎉 SYSTEM READY FOR DETECTION!")
                print("="*60 + "\n")
            else:
                print("❌ Camera not available, cannot start detection")
        else:
            print("❌ ALPR Engine failed to initialize")
            # ALPR failed to initialize
            if hasattr(self, 'status_label'):
                self.status_label.setText("ALPR DISABLED")
                self.status_label.setStyleSheet("color: #ef4444; font-size: 24px; font-weight: 600;")
            
            if hasattr(self, 'info_text'):
                self.info_text.setHtml("""
                    <div style='font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;'>
                        <p style='color: #ef4444; margin: 4px 0;'><b>Error:</b> ALPR Engine failed to initialize</p>
                        <p style='color: #6b7280; margin: 4px 0; font-size: 12px;'>Detection features will be disabled</p>
                    </div>
                """)
    
    def closeEvent(self, event):
        """Handle application close"""
        # Stop initialization threads if running
        if hasattr(self, 'camera_init_thread') and self.camera_init_thread.isRunning():
            self.camera_init_thread.wait(1000)
        if hasattr(self, 'alpr_init_thread') and self.alpr_init_thread.isRunning():
            self.alpr_init_thread.wait(1000)
        
        # Stop detection thread
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.wait()
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Close database
        if self.db:
            self.db.close()
        
        event.accept()


def main():
    # #region agent log
    _log("main_gui.py:main:1", "main() started, creating QApplication", {}, "D")
    # #endregion
    app = QApplication(sys.argv)
    
    # Modern light theme
    # #region agent log
    _log("main_gui.py:main:2", "Setting up theme", {}, "D")
    # #endregion
    app.setStyle('Fusion')
    palette = app.palette()
    palette.setColor(palette.Window, QColor(249, 250, 251))
    palette.setColor(palette.WindowText, QColor(17, 24, 39))
    palette.setColor(palette.Base, QColor(255, 255, 255))
    palette.setColor(palette.AlternateBase, QColor(249, 250, 251))
    palette.setColor(palette.Text, QColor(17, 24, 39))
    palette.setColor(palette.Button, QColor(255, 255, 255))
    palette.setColor(palette.ButtonText, QColor(17, 24, 39))
    palette.setColor(palette.Highlight, QColor(59, 130, 246))
    palette.setColor(palette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    app.setFont(QFont('Segoe UI', 10))
    
    # #region agent log
    _log("main_gui.py:main:3", "Before creating ALPRMainWindow", {}, "D")
    # #endregion
    window = ALPRMainWindow()
    # #region agent log
    _log("main_gui.py:main:4", "After creating ALPRMainWindow, before window.show()", {}, "D")
    # #endregion
    window.show()
    # #region agent log
    _log("main_gui.py:main:5", "After window.show(), before app.exec_()", {}, "D")
    # #endregion
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
