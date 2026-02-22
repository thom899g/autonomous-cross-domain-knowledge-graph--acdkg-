"""
ACDKG Configuration and Firebase Initialization.
Centralized configuration with environment-aware settings and Firebase Admin initialization.
Why: Decouples configuration from business logic, enables environment switching, and ensures 
Firebase is initialized exactly once with proper error handling.
"""
import os
import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
import firebase_admin
from firebase_admin import credentials, firestore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ACDKGConfig:
    """Centralized configuration with type safety and validation."""
    # Data ingestion
    supported_formats: tuple = ('csv', 'json', 'parquet', 'xml')
    max_file_size_mb: int = 100
    api_timeout_seconds: int = 30
    
    # Graph settings
    default_node_collection: str = "knowledge_nodes"
    default_edge_collection: str = "knowledge_edges"
    batch_size: int = 500
    
    # ML settings
    embedding_dimension: int = 128
    similarity_threshold: float = 0.7
    
    # Firebase
    firebase_credential_path: Optional[str] = None
    firestore_project_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_file_size_mb <= 0:
            raise ValueError("max_file_size_mb must be positive")
        if not all(isinstance(fmt, str) for fmt in self.supported_formats):
            raise TypeError("supported_formats must contain strings")
        
        # Try to find Firebase credentials
        if not self.firebase_credential_path:
            self.firebase_credential_path = self._find_firebase_credentials()
    
    def _find_firebase_credentials(self) -> Optional[str]:
        """Search for Firebase credentials in common locations."""
        possible_paths = [
            "serviceAccountKey.json",
            "../serviceAccountKey.json",
            os.path.expanduser("~/.config/firebase/serviceAccountKey.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found Firebase credentials at {path}")
                return path
        
        logger.warning("No Firebase credentials found. Using in-memory mode.")
        return None

class FirebaseManager:
    """Singleton Firebase Admin manager with lazy initialization."""
    _instance: Optional['FirebaseManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._app = None
            self._db = None
            self._config = ACDKGConfig()
            self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase Admin SDK with error handling."""
        if self._config.firebase_credential_path:
            try:
                cred = credentials.Certificate(self._config.firebase_credential_path)
                self._app = firebase_admin.initialize_app(cred)
                self._db = firestore.client()
                logger.info("Firebase Admin SDK initialized successfully")
            except Exception as e:
                logger.error(f"Firebase initialization failed: {e}")
                self._app = None
                self._db = None
        else:
            logger.warning("Firebase not initialized - running in local mode")
    
    @property
    def db(self) -> Optional[firestore.Client]:
        """Get Firestore client with lazy initialization."""
        if self._db is None and not self._initialized:
            self._initialize_firebase()
        return self._db
    
    @property
    def config(self) -> ACDKGConfig:
        return self._config

# Global instances
config = ACDKGConfig()
firebase_manager = FirebaseManager()