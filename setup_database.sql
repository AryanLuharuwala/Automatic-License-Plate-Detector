-- PostgreSQL Database Setup Script for ALPR System

-- Create database
CREATE DATABASE alpr_tollplaza;

-- Connect to database
\c alpr_tollplaza;

-- Create vehicles table
CREATE TABLE vehicles (
    id SERIAL PRIMARY KEY,
    plate_number VARCHAR(20) UNIQUE NOT NULL,
    owner_name VARCHAR(100) NOT NULL,
    vehicle_type VARCHAR(50),
    contact_number VARCHAR(20),
    valid_until DATE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create detection history table
CREATE TABLE detection_history (
    id SERIAL PRIMARY KEY,
    node_id VARCHAR(50) NOT NULL,
    plate_number VARCHAR(20) NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence FLOAT,
    status VARCHAR(20) NOT NULL,
    owner_name VARCHAR(100),
    image_path TEXT
);

-- Create indexes for better performance
CREATE INDEX idx_plate_number ON vehicles(plate_number);
CREATE INDEX idx_detection_timestamp ON detection_history(detected_at);
CREATE INDEX idx_detection_node ON detection_history(node_id);
CREATE INDEX idx_detection_status ON detection_history(status);

-- Insert sample data
INSERT INTO vehicles (plate_number, owner_name, vehicle_type, contact_number, valid_until, notes)
VALUES 
    ('WB-01-AB-1234', 'Rajesh Kumar', 'Car', '9876543210', '2025-12-31', 'Premium member'),
    ('DL-02-CD-5678', 'Priya Sharma', 'SUV', '9876543211', '2025-06-30', 'Monthly pass'),
    ('MH-03-EF-9012', 'Amit Patel', 'Truck', '9876543212', '2026-01-15', 'Commercial vehicle');

-- Create user for application (optional)
CREATE USER alpr_user WITH PASSWORD 'secure_password_here';
GRANT ALL PRIVILEGES ON DATABASE alpr_tollplaza TO alpr_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO alpr_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO alpr_user;

-- Done
SELECT 'Database setup complete!' as status;
