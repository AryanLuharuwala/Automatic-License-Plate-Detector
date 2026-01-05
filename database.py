import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import json
from typing import Optional, List, Dict
import os
import threading

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

class DatabaseManager:
    def __init__(self, config_path: str = "config.json"):
        """Initialize database connection"""
        # #region agent log
        _log("database.py:10", "Loading config for database", {"config_path": config_path}, "A")
        # #endregion
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            # #region agent log
            _log("database.py:11", "Config loaded", {"has_database_key": "database" in config}, "B")
            # #endregion
        except FileNotFoundError as e:
            # #region agent log
            _log("database.py:10", "Config file not found", {"error": str(e)}, "A")
            # #endregion
            raise
        except json.JSONDecodeError as e:
            # #region agent log
            _log("database.py:10", "Config JSON decode error", {"error": str(e)}, "A")
            # #endregion
            raise
        except KeyError as e:
            # #region agent log
            _log("database.py:13", "Missing database config key", {"error": str(e)}, "B")
            # #endregion
            raise
        
        self.db_config = config['database']
        self.conn = None
        self.lock = threading.Lock()  # Thread safety for database operations
        # #region agent log
        _log("database.py:15", "Before database connect", {"conn_is_none": self.conn is None}, "C")
        # #endregion
        self.connect()
        # #region agent log
        _log("database.py:16", "After database connect, before create_tables", {"conn_is_none": self.conn is None}, "C")
        # #endregion
        self.create_tables()
    
    def connect(self):
        """Establish database connection"""
        # #region agent log
        _log("database.py:21", "Before database connection attempt", {"host": self.db_config.get('host'), "port": self.db_config.get('port'), "port_type": type(self.db_config.get('port')).__name__, "password_type": type(self.db_config.get('password')).__name__, "password_length": len(str(self.db_config.get('password', '')))}, "E")
        # #endregion
        try:
            port = self.db_config['port']
            if isinstance(port, str):
                port = int(port)
            # Ensure password is always a string (important for passwords starting with 0 like "0852")
            password = str(self.db_config['password'])
            # #region agent log
            _log("database.py:27", "Password prepared", {"password_type": type(password).__name__, "password_length": len(password)}, "E")
            # #endregion
            self.conn = psycopg2.connect(
                host=self.db_config['host'],
                port=port,
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=password
            )
            # #region agent log
            _log("database.py:28", "Database connection successful", {"conn_is_none": self.conn is None}, "C")
            # #endregion
            print(f"✓ Connected to database: {self.db_config['database']}")
        except Exception as e:
            # #region agent log
            _log("database.py:30", "Database connection failed", {"error": str(e), "error_type": type(e).__name__}, "C")
            # #endregion
            print(f"✗ Database connection failed: {e}")
            raise
    
    def ensure_connection(self):
        """Ensure database connection is alive, reconnect if needed"""
        try:
            if self.conn is None or self.conn.closed:
                # #region agent log
                _log("database.py:ensure_connection:1", "Connection is closed, reconnecting", {}, "C")
                # #endregion
                print("⚠ Database connection lost, reconnecting...")
                self.connect()
                return
            
            # Test if connection is actually alive
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
        except (psycopg2.OperationalError, psycopg2.InterfaceError, AttributeError) as e:
            # #region agent log
            _log("database.py:ensure_connection:2", "Connection test failed, reconnecting", {"error": str(e)}, "C")
            # #endregion
            print(f"⚠ Database connection test failed: {e}, reconnecting...")
            try:
                self.connect()
            except Exception as reconnect_error:
                # #region agent log
                _log("database.py:ensure_connection:3", "Reconnection failed", {"error": str(reconnect_error)}, "C")
                # #endregion
                print(f"✗ Failed to reconnect to database: {reconnect_error}")
                raise
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        # #region agent log
        _log("database.py:35", "Before create_tables", {"conn_is_none": self.conn is None}, "C")
        # #endregion
        if self.conn is None:
            # #region agent log
            _log("database.py:35", "Cannot create tables - conn is None", {}, "C")
            # #endregion
            raise AttributeError("Database connection is None, cannot create tables")
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
        with self.lock:  # Thread-safe access
            try:
                self.ensure_connection()
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
                # #region agent log
                _log("database.py:add_vehicle:1", "Error in add_vehicle", {"error": str(e)}, "J")
                # #endregion
                self.conn.rollback()
                print(f"Error adding vehicle: {e}")
                return False
    
    def get_vehicle(self, plate_number: str) -> Optional[Dict]:
        """Get vehicle details by plate number"""
        print(f"\n💾 [DATABASE] get_vehicle() called for: {plate_number}")
        with self.lock:  # Thread-safe access
            try:
                # #region agent log
                _log("database.py:get_vehicle:1", "Before ensure_connection", {"plate_number": plate_number}, "F")
                # #endregion
                print(f"🔌 [DATABASE] Ensuring connection...")
                self.ensure_connection()
                print(f"✅ [DATABASE] Connection OK")
                # #region agent log
                _log("database.py:get_vehicle:2", "After ensure_connection, before query", {}, "F")
                # #endregion
                
                print(f"🔍 [DATABASE] Executing query for plate: {plate_number.upper()}")
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM vehicles 
                        WHERE plate_number = %s
                    """, (plate_number.upper(),))
                    result = cur.fetchone()
                    # #region agent log
                    _log("database.py:get_vehicle:3", "Query executed", {"result_is_none": result is None}, "F")
                    # #endregion
                    
                    if result:
                        print(f"✅ [DATABASE] Vehicle FOUND: {dict(result)}")
                    else:
                        print(f"❌ [DATABASE] Vehicle NOT FOUND")
                    
                    return dict(result) if result else None
            except Exception as e:
                # #region agent log
                _log("database.py:get_vehicle:4", "Error in get_vehicle", {"error": str(e), "error_type": type(e).__name__}, "F")
                # #endregion
                print(f"❌ [DATABASE] Error fetching vehicle {plate_number}: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    def get_all_vehicles(self) -> List[Dict]:
        """Get all vehicles from database"""
        with self.lock:  # Thread-safe access
            try:
                self.ensure_connection()
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM vehicles 
                        ORDER BY created_at DESC
                    """)
                    results = cur.fetchall()
                    return [dict(row) for row in results]
            except Exception as e:
                # #region agent log
                _log("database.py:get_all_vehicles:1", "Error in get_all_vehicles", {"error": str(e)}, "H")
                # #endregion
                print(f"Error fetching vehicles: {e}")
                return []
    
    def update_vehicle(self, plate_number: str, **kwargs) -> bool:
        """Update vehicle details"""
        with self.lock:  # Thread-safe access
            try:
                self.ensure_connection()
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
        with self.lock:  # Thread-safe access
            try:
                self.ensure_connection()
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
        with self.lock:  # Thread-safe access
            try:
                # #region agent log
                _log("database.py:log_detection:1", "Before ensure_connection", {"plate_number": plate_number, "status": status}, "G")
                # #endregion
                self.ensure_connection()
                
                with self.conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO detection_history 
                        (node_id, plate_number, confidence, status, owner_name, image_path)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (node_id, plate_number.upper(), confidence, status, owner_name, image_path))
                    self.conn.commit()
                    # #region agent log
                    _log("database.py:log_detection:2", "Detection logged successfully", {}, "G")
                    # #endregion
                    return True
            except Exception as e:
                # #region agent log
                _log("database.py:log_detection:3", "Error in log_detection", {"error": str(e), "error_type": type(e).__name__}, "G")
                # #endregion
                self.conn.rollback()
                print(f"Error logging detection: {e}")
                return False
    
    def get_detection_history(self, limit: int = 100) -> List[Dict]:
        """Get recent detection history"""
        with self.lock:  # Thread-safe access
            try:
                self.ensure_connection()
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM detection_history 
                        ORDER BY detected_at DESC 
                        LIMIT %s
                    """, (limit,))
                    results = cur.fetchall()
                    return [dict(row) for row in results]
            except Exception as e:
                # #region agent log
                _log("database.py:get_detection_history:1", "Error in get_detection_history", {"error": str(e)}, "I")
                # #endregion
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
