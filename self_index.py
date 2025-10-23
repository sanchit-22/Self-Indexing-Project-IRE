"""
SelfIndex Implementation - Custom Search Index System

This module implements a custom search index system with multiple variants
following the versioning scheme SelfIndex-v1.xyziq where:
- x: Information indexed (1=Boolean, 2=WordCount, 3=TF-IDF)
- y: Datastore backend (1=Custom, 2=DB1, 3=DB2)
- z: Compression method (1=None, 2=Custom, 3=Library)
- i: Optimization level (0=None, sp=Skipping, th=Threshold, es=EarlyStopping)
- q: Query processing (T=Term-at-a-time, D=Document-at-a-time)
"""

from index_base import IndexBase, IndexInfo, DataStore, Compression, QueryProc, Optimizations
from typing import Iterable, Dict, List, Set, Tuple, Any, Optional
from pathlib import Path
from collections import defaultdict
import json
import pickle
import math
import re
from abc import abstractmethod

# Import query parser
try:
    from query_parser import BooleanQueryExecutor
    QUERY_PARSER_AVAILABLE = True
except ImportError:
    QUERY_PARSER_AVAILABLE = False

# Import compression utilities
try:
    from compression import get_compressor, CompressionBase
    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False

# Import database backends
try:
    from db_backends import get_backend, DatabaseBackend
    DB_BACKENDS_AVAILABLE = True
except ImportError:
    DB_BACKENDS_AVAILABLE = False


class SelfIndex(IndexBase):
    """
    Custom search index implementation with support for multiple variants.
    """
    
    # Class variable for storing all loaded indices
    _loaded_indices: Dict[str, 'SelfIndex'] = {}
    
    # Storage directory for index files
    INDEX_STORAGE_DIR = Path("./indices")
    
    def __init__(self, core='SelfIndex', info='BOOLEAN', dstore='CUSTOM', 
                 qproc='TERMatat', compr='NONE', optim='Null'):
        """
        Initialize a SelfIndex with specified configuration.
        
        Args:
            core: Always 'SelfIndex' for this implementation
            info: Type of information to index (BOOLEAN, WORDCOUNT, TFIDF)
            dstore: Datastore backend (CUSTOM, DB1, DB2)
            qproc: Query processing method (TERMatat, DOCatat)
            compr: Compression method (NONE, CODE, CLIB)
            optim: Optimization strategy (Null, Skipping, Thresholding, EarlyStopping)
        """
        super().__init__(core, info, dstore, qproc, compr, optim)
        
        # Store configuration
        self.info_type = IndexInfo[info]
        self.datastore_type = DataStore[dstore]
        self.compression_type = Compression[compr]
        self.query_proc_type = QueryProc[qproc]
        self.optimization_type = Optimizations[optim]
        
        # Initialize compression handler
        if COMPRESSION_AVAILABLE:
            self.compressor = get_compressor(compr)
        else:
            self.compressor = None
        
        # Initialize database backend
        if DB_BACKENDS_AVAILABLE:
            self.db_backend = get_backend(dstore)
        else:
            self.db_backend = None
        
        # Initialize index structures
        self.inverted_index: Dict[str, Dict[str, Any]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.doc_metadata: Dict[str, Dict[str, Any]] = {}
        self.vocabulary: Set[str] = set()
        self.num_docs: int = 0
        self.avg_doc_length: float = 0.0
        
        # Create storage directory if it doesn't exist
        self.INDEX_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        
    def _get_index_path(self, index_id: str) -> Path:
        """Get the file path for storing an index."""
        return self.INDEX_STORAGE_DIR / f"{index_id}.pkl"
    
    def _get_index_metadata_path(self, index_id: str) -> Path:
        """Get the file path for storing index metadata."""
        return self.INDEX_STORAGE_DIR / f"{index_id}_metadata.json"
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text into tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of preprocessed tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and tokenize
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        
        return tokens
    
    def _build_boolean_index(self, files: Iterable[Tuple[str, str]]) -> None:
        """
        Build a boolean inverted index with document IDs and position IDs.
        
        Args:
            files: Iterable of (file_id, content) tuples
        """
        self.inverted_index.clear()
        self.doc_metadata.clear()
        self.vocabulary.clear()
        self.doc_lengths.clear()
        self.num_docs = 0
        
        for doc_id, content in files:
            tokens = self._preprocess_text(content)
            self.doc_lengths[doc_id] = len(tokens)
            self.doc_metadata[doc_id] = {'length': len(tokens)}
            self.num_docs += 1
            
            # Build posting list with positions
            for position, term in enumerate(tokens):
                self.vocabulary.add(term)
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                if doc_id not in self.inverted_index[term]:
                    self.inverted_index[term][doc_id] = []
                self.inverted_index[term][doc_id].append(position)
        
        # Calculate average document length
        if self.num_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.num_docs
    
    def _build_wordcount_index(self, files: Iterable[Tuple[str, str]]) -> None:
        """
        Build a word count index with term frequencies.
        
        Args:
            files: Iterable of (file_id, content) tuples
        """
        # First build boolean index
        self._build_boolean_index(files)
        
        # Add term frequency information
        for term in self.inverted_index:
            for doc_id in self.inverted_index[term]:
                positions = self.inverted_index[term][doc_id]
                self.inverted_index[term][doc_id] = {
                    'positions': positions,
                    'tf': len(positions)
                }
    
    def _build_tfidf_index(self, files: Iterable[Tuple[str, str]]) -> None:
        """
        Build a TF-IDF index with precomputed TF-IDF weights.
        
        Args:
            files: Iterable of (file_id, content) tuples
        """
        # First build word count index
        self._build_wordcount_index(files)
        
        # Calculate IDF for each term
        idf_scores = {}
        for term in self.inverted_index:
            doc_freq = len(self.inverted_index[term])
            idf_scores[term] = math.log((self.num_docs + 1) / (doc_freq + 1))
        
        # Calculate TF-IDF for each term-document pair
        for term in self.inverted_index:
            for doc_id in self.inverted_index[term]:
                tf = self.inverted_index[term][doc_id]['tf']
                tfidf = tf * idf_scores[term]
                self.inverted_index[term][doc_id]['tfidf'] = tfidf
                self.inverted_index[term][doc_id]['idf'] = idf_scores[term]
    
    def create_index(self, index_id: str, files: Iterable[Tuple[str, str]]) -> None:
        """
        Creates an index for the given files.
        
        Args:
            index_id: The unique identifier for the index
            files: An iterable of tuples (file_id, content)
        """
        # Convert files to list to allow multiple iterations
        files_list = list(files)
        
        # Build appropriate index based on info type
        if self.info_type == IndexInfo.BOOLEAN:
            self._build_boolean_index(files_list)
        elif self.info_type == IndexInfo.WORDCOUNT:
            self._build_wordcount_index(files_list)
        elif self.info_type == IndexInfo.TFIDF:
            self._build_tfidf_index(files_list)
        else:
            raise ValueError(f"Unknown index info type: {self.info_type}")
        
        # Persist index to disk
        self._save_index(index_id)
        
        # Add to loaded indices
        SelfIndex._loaded_indices[index_id] = self
    
    def _save_index(self, index_id: str) -> None:
        """
        Save index to disk based on datastore type.
        
        Args:
            index_id: The unique identifier for the index
        """
        index_data = {
            'inverted_index': self.inverted_index,
            'doc_lengths': self.doc_lengths,
            'doc_metadata': self.doc_metadata,
            'vocabulary': self.vocabulary,
            'num_docs': self.num_docs,
            'avg_doc_length': self.avg_doc_length,
            'identifier_short': self.identifier_short,
            'identifier_long': self.identifier_long
        }
        
        if DB_BACKENDS_AVAILABLE and self.db_backend:
            # Use database backend
            self.db_backend.save_index(index_id, index_data, self.INDEX_STORAGE_DIR)
        else:
            # Fallback to original custom implementation
            index_path = self._get_index_path(index_id)
            with open(index_path, 'wb') as f:
                pickle.dump(index_data, f)
            
            # Save metadata as JSON
            metadata_path = self._get_index_metadata_path(index_id)
            metadata = {
                'index_id': index_id,
                'identifier_short': self.identifier_short,
                'identifier_long': self.identifier_long,
                'num_docs': self.num_docs,
                'vocab_size': len(self.vocabulary)
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_index(self, serialized_index_dump: str) -> None:
        """
        Loads an already created index into memory from disk.
        
        Args:
            serialized_index_dump: Path to dump of serialized index or index_id
        """
        # Check if it's a path or just an index_id
        if '/' in serialized_index_dump or '\\' in serialized_index_dump:
            # It's a path
            index_path = Path(serialized_index_dump)
            
            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")
            
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
        else:
            # It's an index_id
            index_id = serialized_index_dump
            
            if DB_BACKENDS_AVAILABLE and self.db_backend:
                index_data = self.db_backend.load_index(index_id, self.INDEX_STORAGE_DIR)
            else:
                # Fallback to loading from file
                index_path = self._get_index_path(index_id)
                if not index_path.exists():
                    raise FileNotFoundError(f"Index file not found: {index_path}")
                
                with open(index_path, 'rb') as f:
                    index_data = pickle.load(f)
        
        # Restore index structures
        self.inverted_index = index_data['inverted_index']
        self.doc_lengths = index_data['doc_lengths']
        self.doc_metadata = index_data['doc_metadata']
        self.vocabulary = index_data['vocabulary']
        self.num_docs = index_data['num_docs']
        self.avg_doc_length = index_data['avg_doc_length']
    
    def update_index(self, index_id: str, remove_files: Iterable[Tuple[str, str]], 
                     add_files: Iterable[Tuple[str, str]]) -> None:
        """
        Updates an index. First removes files, then adds files.
        
        Args:
            index_id: The unique identifier for the index
            remove_files: Iterable of (file_id, content) tuples to remove
            add_files: Iterable of (file_id, content) tuples to add
        """
        # Remove files
        for doc_id, _ in remove_files:
            self._remove_document(doc_id)
        
        # Add files
        for doc_id, content in add_files:
            self._add_document(doc_id, content)
        
        # Recalculate statistics
        if self.num_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.num_docs
        
        # Save updated index
        self._save_index(index_id)
    
    def _remove_document(self, doc_id: str) -> None:
        """
        Remove a document from the index.
        
        Args:
            doc_id: Document ID to remove
        """
        if doc_id not in self.doc_metadata:
            return
        
        # Remove from inverted index
        terms_to_delete = []
        for term in self.inverted_index:
            if doc_id in self.inverted_index[term]:
                del self.inverted_index[term][doc_id]
                if not self.inverted_index[term]:
                    terms_to_delete.append(term)
        
        # Remove empty terms
        for term in terms_to_delete:
            del self.inverted_index[term]
            self.vocabulary.discard(term)
        
        # Remove metadata
        del self.doc_metadata[doc_id]
        del self.doc_lengths[doc_id]
        self.num_docs -= 1
    
    def _add_document(self, doc_id: str, content: str) -> None:
        """
        Add a document to the index.
        
        Args:
            doc_id: Document ID
            content: Document content
        """
        tokens = self._preprocess_text(content)
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_metadata[doc_id] = {'length': len(tokens)}
        self.num_docs += 1
        
        # Add to inverted index based on index type
        for position, term in enumerate(tokens):
            self.vocabulary.add(term)
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
            
            if self.info_type == IndexInfo.BOOLEAN:
                if doc_id not in self.inverted_index[term]:
                    self.inverted_index[term][doc_id] = []
                self.inverted_index[term][doc_id].append(position)
            else:
                # For WORDCOUNT and TFIDF, we need to recalculate
                if doc_id not in self.inverted_index[term]:
                    self.inverted_index[term][doc_id] = {'positions': [], 'tf': 0}
                self.inverted_index[term][doc_id]['positions'].append(position)
                self.inverted_index[term][doc_id]['tf'] += 1
    
    def query(self, query: str) -> str:
        """
        Queries the loaded index and returns results as JSON string.
        
        Args:
            query: Input query string
            
        Returns:
            JSON string with results
        """
        # Parse and execute query
        results = self._execute_query(query)
        
        # Format results as JSON
        result_dict = {
            'query': query,
            'num_results': len(results),
            'results': results
        }
        
        return json.dumps(result_dict, indent=2)
    
    def _execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a query and return matching documents.
        
        Args:
            query: Query string
            
        Returns:
            List of result documents with scores
        """
        # Check if query contains Boolean operators
        has_boolean_operators = any(op in query.upper() for op in ['AND', 'OR', 'NOT', '"'])
        
        # Use Boolean query parser if available and query has operators
        if QUERY_PARSER_AVAILABLE and has_boolean_operators:
            try:
                executor = BooleanQueryExecutor(self, self.query_proc_type.value + 'ermattime')
                _, results = executor.execute(query)
                return results
            except Exception as e:
                # Fall back to simple query if parsing fails
                print(f"Query parsing failed: {e}. Falling back to simple query.")
                pass
        
        # Choose processing strategy based on query_proc_type
        if self.query_proc_type == QueryProc.DOCatat:
            return self._document_at_a_time_query(query)
        else:
            # Default: Term-at-a-time
            return self._term_at_a_time_query(query)
    
    def _term_at_a_time_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Term-at-a-time query processing.
        Processes one term at a time and accumulates scores.
        
        Args:
            query: Query string
            
        Returns:
            List of ranked results
        """
        terms = self._preprocess_text(query)
        
        # Collect all matching documents
        doc_scores: Dict[str, float] = defaultdict(float)
        
        for term in terms:
            if term in self.inverted_index:
                for doc_id in self.inverted_index[term]:
                    if self.info_type == IndexInfo.BOOLEAN:
                        doc_scores[doc_id] += 1.0
                    elif self.info_type == IndexInfo.WORDCOUNT:
                        doc_scores[doc_id] += self.inverted_index[term][doc_id].get('tf', 1)
                    elif self.info_type == IndexInfo.TFIDF:
                        doc_scores[doc_id] += self.inverted_index[term][doc_id].get('tfidf', 0.0)
        
        # Apply optimizations if enabled
        if self.optimization_type != Optimizations.Null:
            doc_scores = self._apply_optimization(doc_scores, terms)
        
        # Sort by score
        results = [
            {'doc_id': doc_id, 'score': score}
            for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return results
    
    def _document_at_a_time_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Document-at-a-time query processing.
        Processes one document at a time and calculates complete score.
        
        Args:
            query: Query string
            
        Returns:
            List of ranked results
        """
        terms = self._preprocess_text(query)
        
        # Get all candidate documents (documents containing at least one term)
        candidate_docs = set()
        for term in terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term].keys())
        
        # Calculate scores for each document
        doc_scores: Dict[str, float] = {}
        
        for doc_id in candidate_docs:
            score = 0.0
            for term in terms:
                if term in self.inverted_index and doc_id in self.inverted_index[term]:
                    if self.info_type == IndexInfo.BOOLEAN:
                        score += 1.0
                    elif self.info_type == IndexInfo.WORDCOUNT:
                        score += self.inverted_index[term][doc_id].get('tf', 1)
                    elif self.info_type == IndexInfo.TFIDF:
                        score += self.inverted_index[term][doc_id].get('tfidf', 0.0)
            
            doc_scores[doc_id] = score
        
        # Apply optimizations if enabled
        if self.optimization_type != Optimizations.Null:
            doc_scores = self._apply_optimization(doc_scores, terms)
        
        # Sort by score
        results = [
            {'doc_id': doc_id, 'score': score}
            for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return results
    
    def _apply_optimization(self, doc_scores: Dict[str, float], terms: List[str]) -> Dict[str, float]:
        """
        Apply optimization strategy to query results.
        
        Args:
            doc_scores: Dictionary of document scores
            terms: Query terms
            
        Returns:
            Optimized document scores
        """
        if self.optimization_type == Optimizations.Thresholding:
            # Apply threshold-based pruning
            if doc_scores:
                threshold = max(doc_scores.values()) * 0.1  # Keep docs with score >= 10% of max
                doc_scores = {doc_id: score for doc_id, score in doc_scores.items() if score >= threshold}
        
        elif self.optimization_type == Optimizations.EarlyStopping:
            # Early stopping: return top-k results early
            top_k = 100
            if len(doc_scores) > top_k:
                sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                doc_scores = dict(sorted_docs)
        
        elif self.optimization_type == Optimizations.Skipping:
            # Skipping: process only documents with high potential
            # For simplicity, skip documents with very low scores
            if doc_scores:
                avg_score = sum(doc_scores.values()) / len(doc_scores)
                doc_scores = {doc_id: score for doc_id, score in doc_scores.items() if score >= avg_score * 0.5}
        
        return doc_scores
    
    def delete_index(self, index_id: str) -> None:
        """
        Deletes the index with the given index_id.
        
        Args:
            index_id: The unique identifier for the index
        """
        # Remove from loaded indices
        if index_id in SelfIndex._loaded_indices:
            del SelfIndex._loaded_indices[index_id]
        
        if DB_BACKENDS_AVAILABLE and self.db_backend:
            self.db_backend.delete_index(index_id, self.INDEX_STORAGE_DIR)
        else:
            # Fallback to deleting files
            index_path = self._get_index_path(index_id)
            metadata_path = self._get_index_metadata_path(index_id)
            
            if index_path.exists():
                index_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
    
    @classmethod
    def list_indices(cls) -> Iterable[str]:
        """
        Lists all indices.
        
        Returns:
            List of index IDs
        """
        # Try to use database backend
        if DB_BACKENDS_AVAILABLE:
            try:
                # Try each backend type
                for backend_name in ['CUSTOM', 'DB1', 'DB2']:
                    try:
                        backend = get_backend(backend_name)
                        indices = backend.list_indices(cls.INDEX_STORAGE_DIR)
                        if indices:
                            return list(set(indices))  # Remove duplicates
                    except:
                        continue
            except:
                pass
        
        # Fallback to file-based listing
        if not cls.INDEX_STORAGE_DIR.exists():
            return []
        
        indices = []
        for path in cls.INDEX_STORAGE_DIR.glob("*_metadata.json"):
            with open(path, 'r') as f:
                metadata = json.load(f)
                indices.append(metadata['index_id'])
        
        return indices
    
    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        """
        Lists all files indexed in the given index.
        
        Args:
            index_id: The unique identifier for the index
            
        Returns:
            List of file IDs
        """
        return list(self.doc_metadata.keys())
    
    @classmethod
    def load_all_indices(cls) -> Dict[str, 'SelfIndex']:
        """
        Load all available indices from disk.
        
        Returns:
            Dictionary mapping index_id to SelfIndex instance
        """
        loaded = {}
        for index_id in cls.list_indices():
            try:
                index_path = cls.INDEX_STORAGE_DIR / f"{index_id}.pkl"
                metadata_path = cls.INDEX_STORAGE_DIR / f"{index_id}_metadata.json"
                
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Create index instance with same configuration
                # This is simplified - in production would extract config from metadata
                index = cls()
                index.load_index(str(index_path))
                loaded[index_id] = index
                cls._loaded_indices[index_id] = index
            except Exception as e:
                print(f"Failed to load index {index_id}: {e}")
        
        return loaded


# Convenience function for creating indices
def create_self_index(index_id: str, files: Iterable[Tuple[str, str]], 
                      info='BOOLEAN', dstore='CUSTOM', qproc='TERMatat',
                      compr='NONE', optim='Null') -> SelfIndex:
    """
    Convenience function to create a SelfIndex.
    
    Args:
        index_id: Unique identifier for the index
        files: Iterable of (file_id, content) tuples
        info: Index information type
        dstore: Datastore backend
        qproc: Query processing method
        compr: Compression method
        optim: Optimization strategy
        
    Returns:
        Created SelfIndex instance
    """
    index = SelfIndex(info=info, dstore=dstore, qproc=qproc, compr=compr, optim=optim)
    index.create_index(index_id, files)
    return index
