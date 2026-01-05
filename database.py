import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import json
from typing import Optional, List, Dict

class DatabaseManager:
    def __init__(self, config_path: str = "config.json"):
        """Initialize database connection"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.db_config = config['database']
        self.conn = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
            print(f"✓ Connected to database: {self.db_config['database']}")
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            raise
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        with self.conn.cursor() as cur:
            # Vehicles table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vehicles (
                    id SERIAL PRIMARY KEY,
                    plate_number VARCHAR(20) UNIQUE NOT NULL,
                    owner_name VARCHAR(100) NOT NULL,
                    vehicle_type VARCHAR(50),
                    contact_number VARCHAR(20),
                    valid_until DATE,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Detection history table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS detection_history (
                    id SERIAL PRIMARY KEY,
                    node_id VARCHAR(50) NOT NULL,
                    plate_number VARCHAR(20) NOT NULL,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence FLOAT,
                    status VARCHAR(20) NOT NULL,
                    owner_name VARCHAR(100),
                    image_path TEXT
                )
            """)
            
            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_plate_number 
                ON vehicles(plate_number)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_detection_timestamp 
                ON detection_history(detected_at)
            """)
            
            self.conn.commit()
            print("✓ Database tables initialized")
    
    def add_vehicle(self, plate_number: str, owner_name: str, 
                   vehicle_type: str = None, contact_number: str = None,
                   valid_until: str = None, notes: str = None) -> bool:
        """Add a new vehicle to the database"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO vehicles 
                    (plate_number, owner_name, vehicle_type, contact_number, valid_until, notes)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (plate_number.upper(), owner_name, vehicle_type, contact_number, valid_until, notes))
                self.conn.commit()
                return True
        except psycopg2.IntegrityError:
            self.conn.rollback()
            print(f"Vehicle with plate {plate_number} already exists")
            return False
        except Exception as e:
            self.conn.rollback()
            print(f"Error adding vehicle: {e}")
            return False
    
    def get_vehicle(self, plate_number: str) -> Optional[Dict]:
        """Get vehicle details by plate number"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM vehicles 
                    WHERE plate_number = %s
                """, (plate_number.upper(),))
                result = cur.fetchone()
                return dict(result) if result else None
        except Exception as e:
            print(f"Error fetching vehicle: {e}")
            return None
    
    def get_all_vehicles(self) -> List[Dict]:
        """Get all vehicles from database"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM vehicles 
                    ORDER BY created_at DESC
                """)
                results = cur.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            print(f"Error fetching vehicles: {e}")
            return []
    
    def update_vehicle(self, plate_number: str, **kwargs) -> bool:
        """Update vehicle details"""
        try:
            fields = []
            values = []
            for key, value in kwargs.items():
                if value is not None:
                    fields.append(f"{key} = %s")
                    values.append(value)
            
            if not fields:
                return False
            
            values.append(plate_number.upper())
            query = f"UPDATE vehicles SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE plate_number = %s"
            
            with self.conn.cursor() as cur:
                cur.execute(query, values)
                self.conn.commit()
                return True
        except Exception as e:
            self.conn.rollback()
            print(f"Error updating vehicle: {e}")
            return False
    
    def delete_vehicle(self, plate_number: str) -> bool:
        """Delete a vehicle from database"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM vehicles WHERE plate_number = %s", (plate_number.upper(),))
                self.conn.commit()
                return True
        except Exception as e:
            self.conn.rollback()
            print(f"Error deleting vehicle: {e}")
            return False
    
    def log_detection(self, node_id: str, plate_number: str, 
                     confidence: float, status: str, 
                     owner_name: str = None, image_path: str = None) -> bool:
        """Log a detection event"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO detection_history 
                    (node_id, plate_number, confidence, status, owner_name, image_path)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (node_id, plate_number.upper(), confidence, status, owner_name, image_path))
                self.conn.commit()
                return True
        except Exception as e:
            self.conn.rollback()
            print(f"Error logging detection: {e}")
            return False
    
    def get_detection_history(self, limit: int = 100) -> List[Dict]:
        """Get recent detection history"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM detection_history 
                    ORDER BY detected_at DESC 
                    LIMIT %s
                """, (limit,))
                results = cur.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            print(f"Error fetching history: {e}")
            return []
    
    def search_vehicles(self, query: str) -> List[Dict]:
        """Search vehicles by plate number or owner name"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM vehicles 
                    WHERE plate_number ILIKE %s OR owner_name ILIKE %s
                    ORDER BY created_at DESC
                """, (f"%{query}%", f"%{query}%"))
                results = cur.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            print(f"Error searching vehicles: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed")
