import sys
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTabWidget,
                             QTableWidget, QTableWidgetItem, QLineEdit, 
                             QComboBox, QTextEdit, QMessageBox, QHeaderView)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont
import cv2
import numpy as np
from datetime import datetime

from database import DatabaseManager
from camera_handler import CameraHandler
from alpr_engine import ALPREngine

class DetectionThread(QThread):
    """Thread for continuous plate detection"""
    detection_result = pyqtSignal(dict)
    
    def __init__(self, camera, alpr_engine):
        super().__init__()
        self.camera = camera
        self.alpr_engine = alpr_engine
        self.running = False
    
    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                result = self.alpr_engine.process_frame(frame)
                if result:
                    self.detection_result.emit(result)
            self.msleep(100)  # Check every 100ms
    
    def stop(self):
        self.running = False


class ALPRMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALPR Toll Plaza System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Load configuration
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        try:
            self.db = DatabaseManager()
            self.camera = CameraHandler()
            self.alpr_engine = ALPREngine()
            
            self.camera.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", str(e))
            sys.exit(1)
        
        # Detection thread
        self.detection_thread = DetectionThread(self.camera, self.alpr_engine)
        self.detection_thread.detection_result.connect(self.handle_detection)
        
        # UI state
        self.current_detection = None
        
        self.init_ui()
        
        # Start video timer
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video)
        self.video_timer.start(30)  # 30ms = ~33fps
        
        # Start detection
        self.detection_thread.start()
    
    def init_ui(self):
        """Initialize UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Video feed
        left_panel = self.create_video_panel()
        main_layout.addWidget(left_panel, 2)
        
        # Right panel - Tabs
        right_panel = self.create_tabs_panel()
        main_layout.addWidget(right_panel, 1)
    
    def create_video_panel(self):
        """Create video feed panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Video label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #333;")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        # Status panel
        status_layout = QHBoxLayout()
        
        # Detected plate
        self.plate_label = QLabel("NO DETECTION")
        self.plate_label.setFont(QFont('Arial', 24, QFont.Bold))
        self.plate_label.setAlignment(Qt.AlignCenter)
        self.plate_label.setStyleSheet("background-color: #333; color: white; padding: 20px; border-radius: 10px;")
        status_layout.addWidget(self.plate_label)
        
        # Status indicator
        self.status_label = QLabel("WAITING")
        self.status_label.setFont(QFont('Arial', 20, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("background-color: gray; color: white; padding: 20px; border-radius: 10px;")
        status_layout.addWidget(self.status_label)
        
        layout.addLayout(status_layout)
        
        # Vehicle info panel
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        self.info_text.setStyleSheet("background-color: #2a2a2a; color: white; border: 1px solid #333; padding: 10px;")
        layout.addWidget(self.info_text)
        
        return panel
    
    def create_tabs_panel(self):
        """Create tabs panel"""
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; background: #2a2a2a; }
            QTabBar::tab { background: #333; color: white; padding: 10px; }
            QTabBar::tab:selected { background: #0066cc; }
        """)
        
        # Admin tab
        admin_tab = self.create_admin_tab()
        tabs.addTab(admin_tab, "Admin Panel")
        
        # History tab
        history_tab = self.create_history_tab()
        tabs.addTab(history_tab, "History")
        
        # Settings tab
        settings_tab = self.create_settings_tab()
        tabs.addTab(settings_tab, "Settings")
        
        return tabs
    
    def create_admin_tab(self):
        """Create admin panel tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Add vehicle form
        form_layout = QVBoxLayout()
        
        self.plate_input = QLineEdit()
        self.plate_input.setPlaceholderText("Plate Number (e.g., WB-01-AB-1234)")
        form_layout.addWidget(QLabel("Plate Number:"))
        form_layout.addWidget(self.plate_input)
        
        self.owner_input = QLineEdit()
        self.owner_input.setPlaceholderText("Owner Name")
        form_layout.addWidget(QLabel("Owner Name:"))
        form_layout.addWidget(self.owner_input)
        
        self.type_input = QComboBox()
        self.type_input.addItems(['Car', 'SUV', 'Truck', 'Bus', 'Motorcycle'])
        form_layout.addWidget(QLabel("Vehicle Type:"))
        form_layout.addWidget(self.type_input)
        
        self.contact_input = QLineEdit()
        self.contact_input.setPlaceholderText("Contact Number")
        form_layout.addWidget(QLabel("Contact:"))
        form_layout.addWidget(self.contact_input)
        
        add_btn = QPushButton("Add Vehicle")
        add_btn.clicked.connect(self.add_vehicle)
        add_btn.setStyleSheet("background-color: #0066cc; color: white; padding: 10px; font-weight: bold;")
        form_layout.addWidget(add_btn)
        
        layout.addLayout(form_layout)
        
        # Vehicle table
        self.vehicle_table = QTableWidget()
        self.vehicle_table.setColumnCount(5)
        self.vehicle_table.setHorizontalHeaderLabels(['Plate', 'Owner', 'Type', 'Contact', 'Actions'])
        self.vehicle_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.vehicle_table)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.refresh_vehicles)
        layout.addWidget(refresh_btn)
        
        self.refresh_vehicles()
        
        return widget
    
    def create_history_tab(self):
        """Create history tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(['Time', 'Node', 'Plate', 'Status', 'Owner'])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.history_table)
        
        refresh_btn = QPushButton("Refresh History")
        refresh_btn.clicked.connect(self.refresh_history)
        layout.addWidget(refresh_btn)
        
        self.refresh_history()
        
        return widget
    
    def create_settings_tab(self):
        """Create settings tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml(f"""
            <h2>System Information</h2>
            <p><b>Node ID:</b> {self.config['node']['node_id']}</p>
            <p><b>Location:</b> {self.config['node']['location']}</p>
            <p><b>Database:</b> {self.config['database']['host']}:{self.config['database']['port']}</p>
            <p><b>Camera:</b> {self.config['camera']['source']}</p>
            <p><b>YOLO Model:</b> {self.config['yolo']['model_path']}</p>
            <p><b>Device:</b> {self.config['yolo']['device']}</p>
        """)
        layout.addWidget(info_text)
        
        return widget
    
    def update_video(self):
        """Update video feed"""
        ret, frame = self.camera.read()
        if ret:
            # Convert frame to QImage
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
    
    def handle_detection(self, result):
        """Handle detection result from thread"""
        plate_number = result['plate_number']
        confidence = result['confidence']
        
        # Update UI
        self.plate_label.setText(plate_number)
        
        # Check database
        vehicle = self.db.get_vehicle(plate_number)
        
        if vehicle:
            # Vehicle found - ALLOWED
            self.status_label.setText("✓ ALLOWED")
            self.status_label.setStyleSheet(
                "background-color: green; color: white; padding: 20px; border-radius: 10px; font-weight: bold;"
            )
            
            info_html = f"""
                <h3 style='color: #00ff00;'>VEHICLE AUTHORIZED</h3>
                <p><b>Owner:</b> {vehicle['owner_name']}</p>
                <p><b>Type:</b> {vehicle['vehicle_type']}</p>
                <p><b>Contact:</b> {vehicle['contact_number']}</p>
                <p><b>Confidence:</b> {confidence:.2%}</p>
            """
            self.info_text.setHtml(info_html)
            
            # Log detection
            self.db.log_detection(
                self.config['node']['node_id'],
                plate_number,
                confidence,
                'ALLOWED',
                vehicle['owner_name']
            )
        else:
            # Vehicle not found - DENIED
            self.status_label.setText("✗ DENIED")
            self.status_label.setStyleSheet(
                "background-color: red; color: white; padding: 20px; border-radius: 10px; font-weight: bold;"
            )
            
            info_html = f"""
                <h3 style='color: #ff0000;'>VEHICLE NOT AUTHORIZED</h3>
                <p><b>Plate:</b> {plate_number}</p>
                <p><b>Status:</b> Not registered</p>
                <p><b>Confidence:</b> {confidence:.2%}</p>
                <p style='color: yellow;'>Please register at admin panel</p>
            """
            self.info_text.setHtml(info_html)
            
            # Log detection
            self.db.log_detection(
                self.config['node']['node_id'],
                plate_number,
                confidence,
                'DENIED'
            )
        
        # Refresh history
        self.refresh_history()
        
        # Reset after 5 seconds
        QTimer.singleShot(5000, self.reset_display)
    
    def reset_display(self):
        """Reset display to waiting state"""
        self.plate_label.setText("NO DETECTION")
        self.status_label.setText("WAITING")
        self.status_label.setStyleSheet(
            "background-color: gray; color: white; padding: 20px; border-radius: 10px;"
        )
        self.info_text.clear()
    
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
            delete_btn.setStyleSheet("background-color: #cc0000; color: white;")
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
                status_item.setBackground(Qt.green)
            else:
                status_item.setBackground(Qt.red)
            self.history_table.setItem(i, 3, status_item)
            
            self.history_table.setItem(i, 4, QTableWidgetItem(record['owner_name'] or 'Unknown'))
    
    def closeEvent(self, event):
        """Handle application close"""
        self.detection_thread.stop()
        self.detection_thread.wait()
        self.camera.release()
        self.db.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    palette = app.palette()
    palette.setColor(palette.Window, Qt.black)
    palette.setColor(palette.WindowText, Qt.white)
    palette.setColor(palette.Base, Qt.black)
    palette.setColor(palette.AlternateBase, Qt.darkGray)
    palette.setColor(palette.Text, Qt.white)
    palette.setColor(palette.Button, Qt.darkGray)
    palette.setColor(palette.ButtonText, Qt.white)
    app.setPalette(palette)
    
    window = ALPRMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()