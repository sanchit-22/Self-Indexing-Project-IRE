"""
Database backend implementations for SelfIndex.

Implements multiple database backends:
1. Custom - Pickle/JSON (already in SelfIndex)
2. SQLite - Lightweight embedded database
3. Redis - In-memory key-value store (optional, fallback to SQLite if not available)
"""

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


class DatabaseBackend(ABC):
    """Abstract base class for database backends."""
    
    @abstractmethod
    def save_index(self, index_id: str, index_data: Dict[str, Any], storage_dir: Path) -> None:
        """Save index data to the backend."""
        pass
    
    @abstractmethod
    def load_index(self, index_id: str, storage_dir: Path) -> Dict[str, Any]:
        """Load index data from the backend."""
        pass
    
    @abstractmethod
    def delete_index(self, index_id: str, storage_dir: Path) -> None:
        """Delete index data from the backend."""
        pass
    
    @abstractmethod
    def list_indices(self, storage_dir: Path) -> List[str]:
        """List all available indices."""
        pass


class CustomBackend(DatabaseBackend):
    """
    Custom backend using pickle and JSON.
    
    Pros:
    - Simple to implement
    - No external dependencies
    - Fast for small to medium datasets
    - Good for development and testing
    
    Cons:
    - Not scalable to very large datasets
    - No concurrent access support
    - Limited query capabilities
    """
    
    def save_index(self, index_id: str, index_data: Dict[str, Any], storage_dir: Path) -> None:
        """Save index using pickle."""
        storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main index data
        index_path = storage_dir / f"{index_id}.pkl"
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save metadata as JSON
        metadata_path = storage_dir / f"{index_id}_metadata.json"
        metadata = {
            'index_id': index_id,
            'identifier_short': index_data.get('identifier_short', ''),
            'identifier_long': index_data.get('identifier_long', ''),
            'num_docs': index_data.get('num_docs', 0),
            'vocab_size': len(index_data.get('vocabulary', set()))
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_index(self, index_id: str, storage_dir: Path) -> Dict[str, Any]:
        """Load index from pickle."""
        index_path = storage_dir / f"{index_id}.pkl"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        with open(index_path, 'rb') as f:
            return pickle.load(f)
    
    def delete_index(self, index_id: str, storage_dir: Path) -> None:
        """Delete index files."""
        index_path = storage_dir / f"{index_id}.pkl"
        metadata_path = storage_dir / f"{index_id}_metadata.json"
        
        if index_path.exists():
            index_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
    
    def list_indices(self, storage_dir: Path) -> List[str]:
        """List all indices by finding metadata files."""
        if not storage_dir.exists():
            return []
        
        indices = []
        for path in storage_dir.glob("*_metadata.json"):
            with open(path, 'r') as f:
                metadata = json.load(f)
                indices.append(metadata['index_id'])
        
        return indices


class SQLiteBackend(DatabaseBackend):
    """
    SQLite backend using embedded database.
    
    Pros:
    - ACID transactions
    - SQL query support
    - Good performance for medium datasets
    - File-based, no server needed
    - Supports concurrent reads
    - Built-in to Python
    
    Cons:
    - Limited concurrent writes
    - Not suitable for distributed systems
    - Less efficient for very large datasets
    """
    
    def _get_db_path(self, storage_dir: Path) -> Path:
        """Get the database file path."""
        storage_dir.mkdir(parents=True, exist_ok=True)
        return storage_dir / "indices.db"
    
    def _init_db(self, storage_dir: Path) -> None:
        """Initialize database schema."""
        db_path = self._get_db_path(storage_dir)
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create indices table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indices (
                index_id TEXT PRIMARY KEY,
                identifier_short TEXT,
                identifier_long TEXT,
                num_docs INTEGER,
                vocab_size INTEGER,
                index_data BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_index(self, index_id: str, index_data: Dict[str, Any], storage_dir: Path) -> None:
        """Save index to SQLite database."""
        self._init_db(storage_dir)
        
        db_path = self._get_db_path(storage_dir)
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Serialize index data
        serialized_data = pickle.dumps(index_data, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Insert or replace index
        cursor.execute("""
            INSERT OR REPLACE INTO indices 
            (index_id, identifier_short, identifier_long, num_docs, vocab_size, index_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            index_id,
            index_data.get('identifier_short', ''),
            index_data.get('identifier_long', ''),
            index_data.get('num_docs', 0),
            len(index_data.get('vocabulary', set())),
            serialized_data
        ))
        
        conn.commit()
        conn.close()
    
    def load_index(self, index_id: str, storage_dir: Path) -> Dict[str, Any]:
        """Load index from SQLite database."""
        db_path = self._get_db_path(storage_dir)
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT index_data FROM indices WHERE index_id = ?", (index_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row is None:
            raise FileNotFoundError(f"Index not found: {index_id}")
        
        return pickle.loads(row[0])
    
    def delete_index(self, index_id: str, storage_dir: Path) -> None:
        """Delete index from SQLite database."""
        db_path = self._get_db_path(storage_dir)
        
        if not db_path.exists():
            return
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM indices WHERE index_id = ?", (index_id,))
        
        conn.commit()
        conn.close()
    
    def list_indices(self, storage_dir: Path) -> List[str]:
        """List all indices from SQLite database."""
        db_path = self._get_db_path(storage_dir)
        
        if not db_path.exists():
            return []
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT index_id FROM indices ORDER BY index_id")
        rows = cursor.fetchall()
        
        conn.close()
        
        return [row[0] for row in rows]


class RedisBackend(DatabaseBackend):
    """
    Redis backend using in-memory key-value store.
    Falls back to SQLite if Redis is not available.
    
    Pros (Redis):
    - Very fast (in-memory)
    - Good for caching and high-performance scenarios
    - Supports concurrent access
    - Can be distributed
    - Built-in data structures
    
    Cons (Redis):
    - Requires external Redis server
    - Data primarily in memory (can be persisted)
    - More complex deployment
    - Additional dependency
    
    This implementation uses SQLite as fallback for simplicity.
    """
    
    def __init__(self):
        """Initialize Redis backend with SQLite fallback."""
        self.redis_available = False
        self.redis_client = None
        
        # Try to import and connect to Redis
        try:
            import redis
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
            self.redis_client.ping()
            self.redis_available = True
            print("✅ Redis backend available")
        except:
            self.redis_available = False
            print("⚠️  Redis not available, using SQLite fallback")
            self.sqlite_backend = SQLiteBackend()
    
    def save_index(self, index_id: str, index_data: Dict[str, Any], storage_dir: Path) -> None:
        """Save index to Redis or SQLite fallback."""
        if self.redis_available and self.redis_client:
            # Save to Redis
            key = f"index:{index_id}"
            serialized_data = pickle.dumps(index_data, protocol=pickle.HIGHEST_PROTOCOL)
            self.redis_client.set(key, serialized_data)
            
            # Save metadata
            metadata_key = f"index_meta:{index_id}"
            metadata = {
                'index_id': index_id,
                'identifier_short': index_data.get('identifier_short', ''),
                'identifier_long': index_data.get('identifier_long', ''),
                'num_docs': index_data.get('num_docs', 0),
                'vocab_size': len(index_data.get('vocabulary', set()))
            }
            self.redis_client.set(metadata_key, json.dumps(metadata))
        else:
            # Fallback to SQLite
            self.sqlite_backend.save_index(index_id, index_data, storage_dir)
    
    def load_index(self, index_id: str, storage_dir: Path) -> Dict[str, Any]:
        """Load index from Redis or SQLite fallback."""
        if self.redis_available and self.redis_client:
            key = f"index:{index_id}"
            data = self.redis_client.get(key)
            
            if data is None:
                raise FileNotFoundError(f"Index not found: {index_id}")
            
            return pickle.loads(data)
        else:
            return self.sqlite_backend.load_index(index_id, storage_dir)
    
    def delete_index(self, index_id: str, storage_dir: Path) -> None:
        """Delete index from Redis or SQLite fallback."""
        if self.redis_available and self.redis_client:
            key = f"index:{index_id}"
            metadata_key = f"index_meta:{index_id}"
            self.redis_client.delete(key, metadata_key)
        else:
            self.sqlite_backend.delete_index(index_id, storage_dir)
    
    def list_indices(self, storage_dir: Path) -> List[str]:
        """List all indices from Redis or SQLite fallback."""
        if self.redis_available and self.redis_client:
            # List all index metadata keys
            pattern = "index_meta:*"
            keys = self.redis_client.keys(pattern)
            
            indices = []
            for key in keys:
                metadata_str = self.redis_client.get(key)
                if metadata_str:
                    metadata = json.loads(metadata_str)
                    indices.append(metadata['index_id'])
            
            return sorted(indices)
        else:
            return self.sqlite_backend.list_indices(storage_dir)


def get_backend(backend_type: str) -> DatabaseBackend:
    """
    Get a database backend instance.
    
    Args:
        backend_type: 'CUSTOM', 'DB1', or 'DB2'
        
    Returns:
        Database backend instance
    """
    if backend_type == 'CUSTOM':
        return CustomBackend()
    elif backend_type == 'DB1':
        return SQLiteBackend()
    elif backend_type == 'DB2':
        return RedisBackend()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
