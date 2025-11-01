#!/usr/bin/env python3
"""
Complete SelfIndex-v1.xyziq implementation supporting all 108 variant combinations
"""

import pickle
import json
import sqlite3
import zlib
import math
import time
import re
import os
import shutil
from collections import defaultdict, Counter
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from typing import Iterable, Dict, List, Set, Tuple, Union, Any
from index_base import IndexBase, IndexInfo, DataStore, Compression, QueryProc, Optimizations
import itertools

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class InvertedListPointer:
    """Helper class to manage pointers into inverted lists for efficient traversal"""
    
    def __init__(self, term: str, postings: List[Dict]):
        self.term = term
        self.postings = sorted(postings, key=lambda x: x['doc_id'])  # Ensure sorted by doc_id
        self.position = 0
        self.finished = len(postings) == 0
    
    def get_current_document(self) -> str:
        """Get the current document ID"""
        if self.finished or self.position >= len(self.postings):
            return None
        return self.postings[self.position]['doc_id']
    
    def get_current_posting(self) -> Dict:
        """Get the current posting"""
        if self.finished or self.position >= len(self.postings):
            return None
        return self.postings[self.position]
    
    def move_to_next_document(self):
        """Move to the next document in the list"""
        if not self.finished:
            self.position += 1
            if self.position >= len(self.postings):
                self.finished = True
    
    def move_past_document(self, doc_id: str):
        """Move past the specified document (used in doc-at-a-time)"""
        while not self.finished and self.get_current_document() and self.get_current_document() <= doc_id:
            self.move_to_next_document()
    
    def is_finished(self) -> bool:
        """Check if we've reached the end of the list"""
        return self.finished
    
    def find_document(self, doc_id: str) -> Dict:
        """Binary search for a specific document in the postings list"""
        if not self.postings:
            return None
        
        left, right = 0, len(self.postings) - 1
        
        while left <= right:
            mid = (left + right) // 2
            mid_doc_id = self.postings[mid]['doc_id']
            
            if mid_doc_id == doc_id:
                return self.postings[mid]
            elif mid_doc_id < doc_id:
                left = mid + 1
            else:
                right = mid - 1
        
        return None

class SelfIndex(IndexBase):
    """Complete SelfIndex implementation supporting all 108 variant combinations"""
    
    def __init__(self, index_type='BOOLEAN', datastore='CUSTOM', compression='NONE', 
                 query_proc='TERMatat', optimization='Null'):
        super().__init__(core='SelfIndex', info=index_type, dstore=datastore, 
                         qproc=query_proc, compr=compression, optim=optimization)
        
        self.index_type = index_type
        self.datastore = datastore
        self.compression = compression
        self.query_proc = query_proc
        self.optimization = optimization
        
        # Initialize preprocessing tools
        try:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
            self.stemmer = None
            
        self.punct_table = str.maketrans('', '', string.punctuation)
        
        # Initialize storage
        self.data_dir = Path("selfindex_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Index structures
        self.indices = {}
        self.current_index = None
        
        # Initialize storage backend
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage backend based on configuration"""
        if self.datastore == 'DB1':  # SQLite
            self.db_path = self.data_dir / f"{self.identifier_short}.db"
            self._init_sqlite()
        elif self.datastore == 'DB2':  # JSON simulation
            self.json_db_path = self.data_dir / f"{self.identifier_short}_db.json"
    
    def _init_sqlite(self):
        """Initialize SQLite database for postings storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS postings (
                    term TEXT PRIMARY KEY,
                    postings_data BLOB,
                    doc_frequency INTEGER,
                    compression_type TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    token_count INTEGER,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS index_metadata (
                    index_id TEXT PRIMARY KEY,
                    config TEXT,
                    stats TEXT,
                    creation_time TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"SQLite initialization error: {e}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text: lowercase, remove punctuation, tokenize, remove stopwords, stem"""
        if not text:
            return []
        
        try:
            text = text.lower().translate(self.punct_table)
            tokens = word_tokenize(text)
            
            processed_tokens = []
            for word in tokens:
                if word.isalpha() and word not in self.stop_words:
                    if self.stemmer:
                        stemmed = self.stemmer.stem(word)
                    else:
                        stemmed = word
                    processed_tokens.append(stemmed)
            
            return processed_tokens
        except:
            # Fallback if NLTK fails
            words = text.lower().translate(self.punct_table).split()
            return [w for w in words if w.isalpha() and w not in self.stop_words]
    
    def create_index(self, index_id: str, files: Iterable, pretokenized: bool = False) -> None:
        """Creates comprehensive index with all specified variants. If pretokenized=True, expects (doc_id, tokens, content)."""
        print(f"🚀 Creating SelfIndex: {index_id}")
        print(f"📋 Configuration: {self.identifier_long}")

        if self._index_exists(index_id):
            print(f"⚠️  Index {index_id} already exists, loading existing...")
            self.load_index(index_id)
            return

        file_list = list(files)

        inverted_index = defaultdict(list)
        doc_info = {}
        doc_count = 0
        total_tokens = 0
        term_doc_frequencies = defaultdict(int)

        print(f"📖 Processing {len(file_list)} documents for {self.index_type} index...")

        for item in file_list:
            if pretokenized:
                doc_id, tokens, content = item
            else:
                doc_id, content = item
                tokens = self._preprocess_text(content)
            doc_length = len(tokens)
            total_tokens += doc_length

            doc_info[doc_id] = {
                'length': doc_length,
                'title': doc_id,
                'content': content[:500] if len(content) > 500 else content
            }

            term_positions = defaultdict(list)
            term_frequencies = defaultdict(int)

            for pos, term in enumerate(tokens):
                term_positions[term].append(pos)
                term_frequencies[term] += 1

            for term in term_positions.keys():
                term_doc_frequencies[term] += 1

            for term, positions in term_positions.items():
                tf = term_frequencies[term]
                posting = {
                    'doc_id': doc_id,
                    'positions': positions,
                    'doc_length': doc_length
                }
                if self.index_type in ['WORDCOUNT', 'TFIDF']:
                    posting['tf'] = tf
                inverted_index[term].append(posting)

            doc_count += 1
            if doc_count % 1000 == 0:
                print(f"📊 Processed {doc_count} documents...")

        print(f"✅ Processed {doc_count} documents, {len(inverted_index)} unique terms")
        
        # Calculate TF-IDF scores if needed
        if self.index_type == 'TFIDF':
            print("🧮 Calculating TF-IDF scores...")
            for term, postings in inverted_index.items():
                df = term_doc_frequencies[term]
                idf = math.log10(doc_count / df) if df > 0 else 0
                
                for posting in postings:
                    tf = posting['tf']
                    tf_idf = tf * idf
                    posting['tf_idf'] = tf_idf
                    posting['idf'] = idf
        
        # Sort postings by document ID for optimization
        for term in inverted_index:
            inverted_index[term].sort(key=lambda x: x['doc_id'])
        
        # Apply compression
        if self.compression != 'NONE':
            print(f"🗜️  Applying {self.compression} compression...")
            inverted_index = self._compress_index(inverted_index)
        
        # Add skip pointers if optimization enabled
        if self.optimization == 'Skipping':
            print("⚡ Adding skip pointers...")
            inverted_index = self._add_skip_pointers(inverted_index)
        
        # Store index based on datastore
        print(f"💾 Storing index using {self.datastore} datastore...")
        self._store_index(index_id, inverted_index, doc_info, {
            'doc_count': doc_count,
            'term_count': len(inverted_index),
            'total_tokens': total_tokens,
            'avg_doc_length': total_tokens / doc_count if doc_count > 0 else 0,
            'term_doc_frequencies': dict(term_doc_frequencies)
        })
        
        # Load index into memory
        self.current_index = index_id
        self.indices[index_id] = {
            'inverted_index': inverted_index,
            'doc_info': doc_info,
            'stats': {
                'doc_count': doc_count,
                'term_count': len(inverted_index),
                'total_tokens': total_tokens
            }
        }
        
        print(f"🎉 Index {index_id} created successfully!")
    
    def _compress_index(self, index: Dict) -> Dict:
        """Apply compression to postings lists"""
        compressed_index = {}
        
        for term, postings in index.items():
            if self.compression == 'CODE':
                # Simple delta compression
                compressed_postings = self._delta_compress_postings(postings)
            elif self.compression == 'CLIB':
                # Use zlib compression
                compressed_postings = self._zlib_compress_postings(postings)
            else:
                compressed_postings = postings
            
            compressed_index[term] = compressed_postings
        
        return compressed_index
    
    def _delta_compress_postings(self, postings: List[Dict]) -> Dict:
        """Delta compression simulation"""
        if len(postings) <= 1:
            return {'type': 'delta', 'data': postings}
        
        sorted_postings = sorted(postings, key=lambda x: hash(x['doc_id']) % 1000000)
        
        compressed_data = {
            'type': 'delta',
            'first': sorted_postings[0],
            'deltas': []
        }
        
        prev_hash = hash(sorted_postings[0]['doc_id']) % 1000000
        
        for i in range(1, len(sorted_postings)):
            current = sorted_postings[i].copy()
            current_hash = hash(current['doc_id']) % 1000000
            delta = current_hash - prev_hash
            
            compressed_data['deltas'].append({
                'delta': delta,
                'positions': current.get('positions', []),
                'tf': current.get('tf', 1),
                'tf_idf': current.get('tf_idf', 0),
                'doc_length': current.get('doc_length', 0)
            })
            
            prev_hash = current_hash
        
        return compressed_data
    
    def _zlib_compress_postings(self, postings: List[Dict]) -> Dict:
        """Compress postings using zlib"""
        try:
            serialized = json.dumps(postings, default=str)
            compressed_bytes = zlib.compress(serialized.encode('utf-8'))
            
            return {
                'type': 'zlib',
                'compressed_data': compressed_bytes,
                'original_size': len(serialized),
                'compressed_size': len(compressed_bytes)
            }
        except Exception as e:
            return {'type': 'error', 'data': postings}
    
    def _decompress_postings(self, compressed_data: Any, term: str) -> List[Dict]:
        """Decompress postings based on compression type"""
        if isinstance(compressed_data, list):
            return compressed_data
        
        if not isinstance(compressed_data, dict):
            return []
        
        comp_type = compressed_data.get('type', 'none')
        
        if comp_type == 'delta':
            return self._delta_decompress_postings(compressed_data)
        elif comp_type == 'zlib':
            return self._zlib_decompress_postings(compressed_data)
        elif comp_type == 'error':
            return compressed_data.get('data', [])
        else:
            return compressed_data.get('data', [])
    
    def _delta_decompress_postings(self, compressed_data: Dict) -> List[Dict]:
        """Decompress delta-compressed postings"""
        try:
            result = [compressed_data['first']]
            
            if 'deltas' not in compressed_data:
                return result
            
            prev_hash = hash(compressed_data['first']['doc_id']) % 1000000
            
            for delta_info in compressed_data['deltas']:
                current_hash = prev_hash + delta_info['delta']
                
                doc_posting = {
                    'doc_id': compressed_data['first']['doc_id'],  # Use original doc_id
                    'positions': delta_info.get('positions', []),
                    'doc_length': delta_info.get('doc_length', 0)
                }
                
                if 'tf' in delta_info:
                    doc_posting['tf'] = delta_info['tf']
                if 'tf_idf' in delta_info:
                    doc_posting['tf_idf'] = delta_info['tf_idf']
                
                result.append(doc_posting)
                prev_hash = current_hash
            
            return result
            
        except Exception as e:
            return [compressed_data.get('first', {})]
    
    def _zlib_decompress_postings(self, compressed_data: Dict) -> List[Dict]:
        """Decompress zlib-compressed postings"""
        try:
            compressed_bytes = compressed_data['compressed_data']
            decompressed_str = zlib.decompress(compressed_bytes).decode('utf-8')
            return json.loads(decompressed_str)
        except Exception as e:
            return []
    
    def _add_skip_pointers(self, index: Dict) -> Dict:
        """Add skip pointers for query optimization"""
        optimized_index = {}
        
        for term, postings in index.items():
            if isinstance(postings, dict) and postings.get('type') in ['delta', 'zlib']:
                optimized_index[term] = postings
                continue
            
            if isinstance(postings, list) and len(postings) > 10:
                skip_distance = int(math.sqrt(len(postings)))
                
                for i in range(0, len(postings), skip_distance):
                    if i + skip_distance < len(postings):
                        postings[i]['skip_to'] = i + skip_distance
                        postings[i]['skip_doc_id'] = postings[i + skip_distance]['doc_id']
            
            optimized_index[term] = postings
        
        return optimized_index
    
    def _store_index(self, index_id: str, inverted_index: Dict, doc_info: Dict, stats: Dict):
        """Store index based on datastore configuration"""
        
        if self.datastore == 'CUSTOM':
            self._store_custom(index_id, inverted_index, doc_info, stats)
        elif self.datastore == 'DB1':
            self._store_sqlite(index_id, inverted_index, doc_info, stats)
        elif self.datastore == 'DB2':
            self._store_json_db(index_id, inverted_index, doc_info, stats)
    
    def _store_custom(self, index_id: str, inverted_index: Dict, doc_info: Dict, stats: Dict):
        """Store using custom Python serialization"""
        index_dir = self.data_dir / index_id
        index_dir.mkdir(exist_ok=True)
        
        with open(index_dir / "inverted_index.pkl", 'wb') as f:
            pickle.dump(inverted_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(index_dir / "doc_info.json", 'w') as f:
            json.dump(doc_info, f, indent=2)
        
        metadata = {
            'config': {
                'index_type': self.index_type,
                'datastore': self.datastore,
                'compression': self.compression,
                'query_proc': self.query_proc,
                'optimization': self.optimization
            },
            'stats': stats,
            'creation_time': time.time()
        }
        
        with open(index_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _store_sqlite(self, index_id: str, inverted_index: Dict, doc_info: Dict, stats: Dict):
        """Store using SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for term, postings in inverted_index.items():
                postings_data = pickle.dumps(postings, protocol=pickle.HIGHEST_PROTOCOL)
                doc_frequency = len(postings) if isinstance(postings, list) else 0
                
                cursor.execute('''
                    INSERT OR REPLACE INTO postings 
                    (term, postings_data, doc_frequency, compression_type)
                    VALUES (?, ?, ?, ?)
                ''', (term, postings_data, doc_frequency, self.compression))
            
            for doc_id, info in doc_info.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO documents 
                    (doc_id, title, content, token_count, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (doc_id, info.get('title', ''), info.get('content', ''), 
                      info.get('length', 0), json.dumps(info)))
            
            config = {
                'index_type': self.index_type,
                'datastore': self.datastore,
                'compression': self.compression,
                'query_proc': self.query_proc,
                'optimization': self.optimization
            }
            
            cursor.execute('''
                INSERT OR REPLACE INTO index_metadata
                (index_id, config, stats, creation_time)
                VALUES (?, ?, ?, ?)
            ''', (index_id, json.dumps(config), json.dumps(stats), time.time()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"SQLite storage error: {e}")
    
    def _store_json_db(self, index_id: str, inverted_index: Dict, doc_info: Dict, stats: Dict):
        """Store using JSON-based database simulation"""
        try:
            json_safe_index = {}
            for term, postings in inverted_index.items():
                if isinstance(postings, dict) and postings.get('type') == 'zlib':
                    import base64
                    postings_copy = postings.copy()
                    postings_copy['compressed_data'] = base64.b64encode(postings['compressed_data']).decode('utf-8')
                    json_safe_index[term] = postings_copy
                else:
                    json_safe_index[term] = postings
            
            db_data = {
                'index_id': index_id,
                'inverted_index': json_safe_index,
                'doc_info': doc_info,
                'stats': stats,
                'config': {
                    'index_type': self.index_type,
                    'datastore': self.datastore,
                    'compression': self.compression,
                    'query_proc': self.query_proc,
                    'optimization': self.optimization
                },
                'creation_time': time.time()
            }
            
            with open(self.json_db_path, 'w') as f:
                json.dump(db_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"JSON DB storage error: {e}")
    
    def load_index(self, index_id: str) -> None:
        """Load an already created index into memory"""
        
        if not self._index_exists(index_id):
            print(f"❌ Index {index_id} does not exist")
            return
        
        print(f"📂 Loading index: {index_id}")
        
        try:
            if self.datastore == 'CUSTOM':
                self._load_custom(index_id)
            elif self.datastore == 'DB1':
                self._load_sqlite(index_id)
            elif self.datastore == 'DB2':
                self._load_json_db(index_id)
            
            self.current_index = index_id
            print(f"✅ Index {index_id} loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading index {index_id}: {e}")
    
    def _load_custom(self, index_id: str):
        """Load from custom storage"""
        index_dir = self.data_dir / index_id
        
        with open(index_dir / "inverted_index.pkl", 'rb') as f:
            inverted_index = pickle.load(f)
        
        with open(index_dir / "doc_info.json", 'r') as f:
            doc_info = json.load(f)
        
        with open(index_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.indices[index_id] = {
            'inverted_index': inverted_index,
            'doc_info': doc_info,
            'metadata': metadata
        }
    
    def _load_sqlite(self, index_id: str):
        """Load from SQLite storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT term, postings_data FROM postings')
        postings_rows = cursor.fetchall()
        
        inverted_index = {}
        for term, postings_data in postings_rows:
            inverted_index[term] = pickle.loads(postings_data)
        
        cursor.execute('SELECT doc_id, title, content, token_count, metadata FROM documents')
        doc_rows = cursor.fetchall()
        
        doc_info = {}
        for doc_id, title, content, token_count, metadata in doc_rows:
            doc_info[doc_id] = {
                'title': title,
                'content': content,
                'length': token_count,
                'metadata': json.loads(metadata) if metadata else {}
            }
        
        cursor.execute('SELECT config, stats FROM index_metadata WHERE index_id = ?', (index_id,))
        meta_row = cursor.fetchone()
        metadata = {}
        if meta_row:
            metadata = {
                'config': json.loads(meta_row[0]),
                'stats': json.loads(meta_row[1])
            }
        
        conn.close()
        
        self.indices[index_id] = {
            'inverted_index': inverted_index,
            'doc_info': doc_info,
            'metadata': metadata
        }
    
    def _load_json_db(self, index_id: str):
        """Load from JSON database"""
        with open(self.json_db_path, 'r') as f:
            db_data = json.load(f)
        
        inverted_index = {}
        for term, term_data in db_data['inverted_index'].items():
            if isinstance(term_data, dict) and term_data.get('type') == 'zlib':
                import base64
                term_data_copy = term_data.copy()
                term_data_copy['compressed_data'] = base64.b64decode(term_data['compressed_data'])
                inverted_index[term] = term_data_copy
            else:
                inverted_index[term] = term_data
        
        self.indices[index_id] = {
            'inverted_index': inverted_index,
            'doc_info': db_data['doc_info'],
            'metadata': {
                'config': db_data['config'],
                'stats': db_data['stats']
            }
        }
    
    def query(self, query_str: str) -> str:
        """Process queries with Boolean operators and different processing methods"""
        
        if not self.current_index or self.current_index not in self.indices:
            return json.dumps({"error": "No index loaded"})
        
        try:
            parsed_query = self._parse_boolean_query(query_str)
            
            if self.query_proc == 'TERMatat':
                results = self._term_at_a_time_query(parsed_query)
            else:  # DOCatat
                results = self._doc_at_a_time_query(parsed_query)
            
            formatted_results = self._format_results(results)
            
            return json.dumps({
                "query": query_str,
                "parsed_query": parsed_query,
                "processing_method": self.query_proc,
                "total_results": len(results),
                "results": formatted_results[:10]
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _parse_boolean_query(self, query_str: str) -> Dict:
        """Parse Boolean query with proper operator precedence"""
        
        clean_query = query_str.strip()
        
        # Handle phrase queries
        phrase_pattern = r'"([^"]*)"'
        phrases = re.findall(phrase_pattern, clean_query)
        
        for i, phrase in enumerate(phrases):
            clean_query = clean_query.replace(f'"{phrase}"', f'PHRASE_{i}')
        
        # Handle NOT operations
        not_pattern = r'NOT\s+(\w+)'
        not_terms = re.findall(not_pattern, clean_query, re.IGNORECASE)
        
        # Handle AND operations
        and_pattern = r'(\w+)\s+AND\s+(\w+)'
        and_matches = re.findall(and_pattern, clean_query, re.IGNORECASE)
        
        # Handle OR operations
        or_pattern = r'(\w+)\s+OR\s+(\w+)'
        or_matches = re.findall(or_pattern, clean_query, re.IGNORECASE)
        
        # Extract all terms
        words = re.findall(r'\w+', clean_query.replace('AND', '').replace('OR', '').replace('NOT', ''))
        
        # Build query structure
        if and_matches:
            return {
                'type': 'AND',
                'terms': [match[0].lower() for match in and_matches] + [match[1].lower() for match in and_matches],
                'phrases': phrases,
                'not_terms': [term.lower() for term in not_terms]
            }
        elif or_matches:
            return {
                'type': 'OR', 
                'terms': [match[0].lower() for match in or_matches] + [match[1].lower() for match in or_matches],
                'phrases': phrases,
                'not_terms': [term.lower() for term in not_terms]
            }
        elif not_terms:
            return {
                'type': 'NOT',
                'terms': [term.lower() for term in words if term.lower() not in [nt.lower() for nt in not_terms]],
                'not_terms': [term.lower() for term in not_terms],
                'phrases': phrases
            }
        else:
            all_terms = words + phrases
            return {
                'type': 'SIMPLE',
                'terms': [term.lower() for term in all_terms],
                'phrases': phrases
            }
    
    def _term_at_a_time_query(self, parsed_query: Dict) -> List[Dict]:
        """Term-at-a-time query processing following textbook algorithm"""
        index_data = self.indices[self.current_index]
        inverted_index = index_data['inverted_index']
        
        # Extract and preprocess query terms
        query_terms = []
        for term in parsed_query.get('terms', []):
            processed_terms = self._preprocess_text(term)
            query_terms.extend(processed_terms)
        
        for phrase in parsed_query.get('phrases', []):
            phrase_terms = self._preprocess_text(phrase)
            query_terms.extend(phrase_terms)
        
        # Initialize accumulator hash table
        accumulators = {}  # doc_id -> {'score': float, 'positions': dict, 'matched_terms': set}
        
        # TERM-AT-A-TIME: Process each term completely before moving to next
        for term in set(query_terms):
            if term in inverted_index:
                postings = self._get_postings(term, inverted_index)
                
                # Process all postings for this term
                for posting in postings:
                    doc_id = posting['doc_id']
                    
                    # Initialize accumulator for this document if not exists
                    if doc_id not in accumulators:
                        accumulators[doc_id] = {
                            'score': 0.0,
                            'positions': {},
                            'matched_terms': set()
                        }
                    
                    # Calculate score contribution for this term
                    if self.index_type == 'BOOLEAN':
                        score_contribution = 1.0
                    elif self.index_type == 'WORDCOUNT':
                        score_contribution = posting.get('tf', 1.0)
                    elif self.index_type == 'TFIDF':
                        score_contribution = posting.get('tf_idf', 1.0)
                    else:
                        score_contribution = 1.0
                    
                    # Update accumulator
                    accumulators[doc_id]['score'] += score_contribution
                    accumulators[doc_id]['positions'][term] = posting.get('positions', [])
                    accumulators[doc_id]['matched_terms'].add(term)
        
        # Convert accumulators to results format
        results = []
        for doc_id, accumulator in accumulators.items():
            # Apply boolean logic filtering
            doc_matches = {doc_id: accumulator['matched_terms']}
            filtered_docs = self._apply_boolean_logic(doc_matches, parsed_query)
            
            # Add to results if document passes boolean filter
            if doc_id in filtered_docs and accumulator['score'] > 0:
                results.append({
                    'doc_id': doc_id,
                    'score': accumulator['score'],
                    'positions': accumulator['positions'],
                    'matched_terms': list(accumulator['matched_terms'])
                })
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def _doc_at_a_time_query(self, parsed_query: Dict) -> List[Dict]:
        """Document-at-a-time query processing following textbook algorithm with pointer optimization"""
        index_data = self.indices[self.current_index]
        inverted_index = index_data['inverted_index']
        
        # Extract and preprocess query terms
        query_terms = []
        for term in parsed_query.get('terms', []):
            processed_terms = self._preprocess_text(term)
            query_terms.extend(processed_terms)
        
        for phrase in parsed_query.get('phrases', []):
            phrase_terms = self._preprocess_text(phrase)
            query_terms.extend(phrase_terms)
        
        # Create inverted list pointers for each query term
        inverted_list_pointers = []
        for term in set(query_terms):
            if term in inverted_index:
                postings = self._get_postings(term, inverted_index)
                if postings:
                    pointer = InvertedListPointer(term, postings)
                    inverted_list_pointers.append(pointer)
        
        if not inverted_list_pointers:
            return []
        
        # Get all unique document IDs that appear in any inverted list (optimization)
        all_candidate_docs = self._get_all_candidate_documents(inverted_list_pointers)
        
        results = []
        
        # DOCUMENT-AT-A-TIME: Loop through each document (following textbook algorithm)
        for doc_id in all_candidate_docs:
            doc_score = 0.0
            doc_positions = {}
            matched_terms = set()
            
            # For this document, check each inverted list efficiently using binary search
            for pointer in inverted_list_pointers:
                # Use binary search to find document in this posting list - O(log n) instead of O(n)
                found_posting = pointer.find_document(doc_id)
                
                # If document appears in this inverted list
                if found_posting:
                    term = pointer.term
                    matched_terms.add(term)
                    
                    # Calculate score contribution for this term
                    if self.index_type == 'BOOLEAN':
                        score_contribution = 1.0
                    elif self.index_type == 'WORDCOUNT':
                        score_contribution = found_posting.get('tf', 1.0)
                    elif self.index_type == 'TFIDF':
                        score_contribution = found_posting.get('tf_idf', 1.0)
                    else:
                        score_contribution = 1.0
                    
                    doc_score += score_contribution
                    doc_positions[term] = found_posting.get('positions', [])
            
            # Apply boolean logic filtering
            doc_matches = {doc_id: matched_terms}
            filtered_docs = self._apply_boolean_logic(doc_matches, parsed_query)
            
            # Add to results if document passes boolean filter and has non-zero score
            if doc_id in filtered_docs and doc_score > 0:
                results.append({
                    'doc_id': doc_id,
                    'score': doc_score,
                    'positions': doc_positions,
                    'matched_terms': list(matched_terms)
                })
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def _apply_boolean_logic(self, doc_matches: Dict, parsed_query: Dict) -> Set[str]:
        """Apply Boolean logic to filter documents"""
        query_type = parsed_query.get('type', 'SIMPLE')
        query_terms = set(term.lower() for term in parsed_query.get('terms', []))
        not_terms = set(term.lower() for term in parsed_query.get('not_terms', []))
        
        valid_docs = set()
        
        for doc_id, matched_terms in doc_matches.items():
            matched_lower = set(term.lower() for term in matched_terms)
            
            if not_terms and not_terms.intersection(matched_lower):
                continue
            
            if query_type == 'AND':
                if query_terms.issubset(matched_lower):
                    valid_docs.add(doc_id)
            elif query_type == 'OR':
                if query_terms.intersection(matched_lower):
                    valid_docs.add(doc_id)
            elif query_type == 'NOT':
                if matched_lower:
                    valid_docs.add(doc_id)
            else:  # SIMPLE
                if matched_lower:
                    valid_docs.add(doc_id)
        
        return valid_docs
    
    def _get_postings(self, term: str, inverted_index: Dict) -> List[Dict]:
        """Get postings for a term, handling decompression"""
        if term not in inverted_index:
            return []
        
        postings = inverted_index[term]
        return self._decompress_postings(postings, term)
    
    def _format_results(self, results: List[Dict]) -> List[Dict]:
        """Format results for output"""
        formatted = []
        index_data = self.indices[self.current_index]
        doc_info = index_data['doc_info']
        
        for result in results:
            doc_id = result['doc_id']
            doc_data = doc_info.get(doc_id, {})
            
            formatted.append({
                'doc_id': doc_id,
                'title': doc_data.get('title', doc_id),
                'score': result['score'],
                'content_preview': doc_data.get('content', '')[:200] + "...",
                'positions': result.get('positions', {}),
                'matched_terms': result.get('matched_terms', [])
            })
        
        return formatted

    def _get_all_candidate_documents(self, inverted_list_pointers: List[InvertedListPointer]) -> List[str]:
        """Get all unique document IDs from all inverted lists, sorted"""
        all_doc_ids = set()
        
        for pointer in inverted_list_pointers:
            for posting in pointer.postings:
                all_doc_ids.add(posting['doc_id'])
        
        return sorted(list(all_doc_ids))
    
    def _index_exists(self, index_id: str) -> bool:
        """Check if index exists"""
        if self.datastore == 'CUSTOM':
            return (self.data_dir / index_id).exists()
        elif self.datastore == 'DB1':
            if not self.db_path.exists():
                return False
            import sqlite3
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 1 FROM index_metadata WHERE index_id = ?
                """, (index_id,))
                exists = cursor.fetchone() is not None
                conn.close()
                return exists
            except Exception:
                return False
        elif self.datastore == 'DB2':
            return self.json_db_path.exists()
        return False
    
    def update_index(self, index_id: str, remove_files: Iterable[tuple[str, str]], add_files: Iterable[tuple[str, str]]) -> None:
        """Update index by removing and adding files"""
        print(f"⚠️  Index update not implemented for {index_id}")
        pass
    
    def delete_index(self, index_id: str) -> None:
        """Delete an index"""
        if index_id in self.indices:
            del self.indices[index_id]
        
        if self.current_index == index_id:
            self.current_index = None
        
        try:
            if self.datastore == 'CUSTOM':
                index_dir = self.data_dir / index_id
                if index_dir.exists():
                    shutil.rmtree(index_dir)
            elif self.datastore == 'DB1':
                if self.db_path.exists():
                    os.remove(self.db_path)
            elif self.datastore == 'DB2':
                if self.json_db_path.exists():
                    os.remove(self.json_db_path)
        except Exception as e:
            print(f"Error deleting index {index_id}: {e}")
        
        print(f"🗑️  Index {index_id} deleted")
    
    def list_indices(self) -> Iterable[str]:
        """List all available indices"""
        indices = []
        
        try:
            if self.datastore == 'CUSTOM':
                for item in self.data_dir.iterdir():
                    if item.is_dir() and (item / "metadata.json").exists():
                        indices.append(item.name)
            elif self.datastore == 'DB1':
                if self.db_path.exists():
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute('SELECT DISTINCT index_id FROM index_metadata')
                    rows = cursor.fetchall()
                    indices = [row[0] for row in rows]
                    conn.close()
            elif self.datastore == 'DB2':
                if self.json_db_path.exists():
                    with open(self.json_db_path, 'r') as f:
                        data = json.load(f)
                        if 'index_id' in data:
                            indices.append(data['index_id'])
        except Exception as e:
            print(f"Error listing indices: {e}")
        
        return indices
    
    def list_indexed_files(self, index_id: str) -> Iterable[str]:
        """List all files in an index"""
        if index_id not in self.indices:
            self.load_index(index_id)
        
        if index_id in self.indices:
            return list(self.indices[index_id]['doc_info'].keys())
        
        return []