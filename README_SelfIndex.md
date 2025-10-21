# SelfIndex-v1.0 Implementation

A custom search index system with support for multiple variants of indexing, query processing, and compression.

## Overview

SelfIndex is a from-scratch implementation of an inverted index system that supports:

- **Multiple index types**: Boolean, WordCount, and TF-IDF
- **Boolean query processing**: Support for AND, OR, NOT, and PHRASE queries with proper precedence
- **Compression**: Custom gap+varbyte encoding and zlib compression
- **Persistence**: Save and load indices from disk
- **Extensible architecture**: Easy to add new index types and query processors

## Features

### Index Types (x parameter)

1. **Boolean Index (x=1)**: Basic inverted index with document IDs and position information
   - Stores term → {doc_id → [positions]} mappings
   - Supports positional phrase queries

2. **WordCount Index (x=2)**: Extended index with term frequency information
   - Stores term frequencies for ranking
   - Better ranking than Boolean index

3. **TF-IDF Index (x=3)**: Advanced index with TF-IDF weighting
   - Precomputes TF-IDF scores for each term-document pair
   - Best ranking quality for text search

### Query Processing

#### Boolean Query Grammar

The query parser supports full Boolean query syntax:

```
QUERY    := EXPR
EXPR     := TERM | (EXPR) | EXPR AND EXPR | EXPR OR EXPR | NOT EXPR | PHRASE
TERM     := a single word surrounded with double quotes, e.g., "apple"
PHRASE   := multiple words surrounded with double quotes, e.g., "quick brown fox"
```

**Operator Precedence** (highest to lowest):
1. PHRASE
2. NOT
3. AND
4. OR

#### Example Queries

```python
# Simple term search
"apple"

# Boolean operators
"apple" AND "banana"
"apple" OR "orange"
NOT "grape"

# Complex queries with precedence
("apple" OR "banana") AND NOT "cherry"

# Phrase queries
"quick brown fox"
"machine learning" AND "deep learning"
```

### Compression Methods (z parameter)

1. **No Compression (z=NONE)**: Stores data as-is using pickle
   
2. **Custom Compression (z=CODE)**: 
   - Gap encoding: Stores differences between consecutive positions
   - Variable-byte encoding: Uses variable-length byte sequences
   - Achieves ~36% of original size for sequential positions
   
3. **Library Compression (z=CLIB)**:
   - Uses zlib compression
   - Achieves ~68% of original size
   - Faster but less effective than custom compression

### Datastore Backends (y parameter)

Currently implemented:
- **Custom Backend (y=CUSTOM)**: Uses pickle and JSON for persistence

Planned:
- PostgreSQL with GIN indices (y=DB1)
- RocksDB or Redis (y=DB2)

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from self_index import create_self_index

# Create documents
documents = [
    ("doc1", "The quick brown fox jumps over the lazy dog"),
    ("doc2", "A quick brown dog runs fast"),
    ("doc3", "The lazy cat sleeps all day"),
]

# Create a Boolean index
index = create_self_index(
    index_id='my_index',
    files=documents,
    info='BOOLEAN',  # Index type
    dstore='CUSTOM', # Datastore
    qproc='TERMatat', # Query processing
    compr='NONE',    # Compression
    optim='Null'     # Optimizations
)

# Query the index
result = index.query('"quick" AND "brown"')
print(result)
```

### Advanced Boolean Queries

```python
# Create index
index = create_self_index('advanced', documents, info='BOOLEAN')

# AND query
result = index.query('"apple" AND "banana"')

# OR query
result = index.query('"apple" OR "orange"')

# NOT query
result = index.query('NOT "grape"')

# Complex query with parentheses
result = index.query('("apple" OR "banana") AND NOT "cherry"')

# Phrase query
result = index.query('"quick brown fox"')
```

### Working with Different Index Types

```python
# Boolean Index
bool_idx = create_self_index('bool', docs, info='BOOLEAN')

# WordCount Index (with term frequencies)
wc_idx = create_self_index('wc', docs, info='WORDCOUNT')

# TF-IDF Index (with precomputed scores)
tfidf_idx = create_self_index('tfidf', docs, info='TFIDF')

# Compare results
query = "search term"
print("Boolean:", bool_idx.query(query))
print("WordCount:", wc_idx.query(query))
print("TF-IDF:", tfidf_idx.query(query))
```

### Using Compression

```python
# No compression
idx_none = create_self_index('idx1', docs, compr='NONE')

# Custom compression (gap + varbyte)
idx_custom = create_self_index('idx2', docs, compr='CODE')

# Library compression (zlib)
idx_lib = create_self_index('idx3', docs, compr='CLIB')
```

### Index Persistence

```python
# Indices are automatically saved to disk
index = create_self_index('persistent', docs, info='BOOLEAN')

# List all indices
all_indices = SelfIndex.list_indices()
print(f"Available indices: {list(all_indices)}")

# Load an existing index
loaded_index = SelfIndex(info='BOOLEAN', dstore='CUSTOM')
loaded_index.load_index('indices/persistent.pkl')

# List files in an index
files = loaded_index.list_indexed_files('persistent')
print(f"Indexed files: {list(files)}")
```

### Updating Indices

```python
# Add documents
new_docs = [("doc4", "New document content")]
index.update_index('my_index', remove_files=[], add_files=new_docs)

# Remove documents
remove_docs = [("doc1", "")]
index.update_index('my_index', remove_files=remove_docs, add_files=[])
```

## Running Examples

```bash
# Run the example usage script
python example_usage.py
```

This will demonstrate:
- Basic index creation and querying
- Boolean query operations
- Phrase queries
- Index persistence
- Index updates
- All index variants

## Running Tests

```bash
# Run all tests
python -m unittest discover -v

# Run specific test suites
python -m unittest test_self_index -v
python -m unittest test_query_parser -v
python -m unittest test_compression -v
```

Test coverage:
- 19 tests for core SelfIndex functionality
- 21 tests for query parser and execution
- 20 tests for compression methods
- **Total: 60 tests, all passing**

## Architecture

### Core Components

1. **self_index.py**: Main SelfIndex class implementing IndexBase interface
   - Index creation and management
   - Text preprocessing
   - Query execution
   - Persistence

2. **query_parser.py**: Boolean query parser and executor
   - Tokenization
   - Parsing with proper precedence
   - AST evaluation
   - Phrase query support

3. **compression.py**: Compression utilities
   - Custom gap + variable-byte encoding
   - zlib compression
   - Pluggable compression interface

4. **index_base.py**: Abstract base class defining the index interface

### Design Patterns

- **Strategy Pattern**: Compression methods are pluggable strategies
- **Template Method**: IndexBase defines the interface template
- **Builder Pattern**: `create_self_index()` convenience function
- **Factory Pattern**: `get_compressor()` creates appropriate compressor

## Versioning Scheme

Index variants use the scheme `SelfIndex-v1.xyziq`:

- **x**: Index information type (1=Boolean, 2=WordCount, 3=TF-IDF)
- **y**: Datastore backend (1=Custom, 2=DB1, 3=DB2)
- **z**: Compression method (1=None, 2=Custom, 3=Library)
- **i**: Optimization level (0=None, sp=Skipping, th=Threshold, es=EarlyStopping)
- **q**: Query processing (T=Term-at-a-time, D=Document-at-a-time)

Example: `SelfIndex_i1d1c2qTo0` = Boolean index, Custom datastore, Custom compression, Term-at-a-time query processing, No optimizations

## Performance

### Compression Ratios

Based on 1000 sequential positions:
- No compression: 2760 bytes (100%)
- Custom compression: 1000 bytes (36%)
- Library compression: 1880 bytes (68%)

Custom compression performs best for posting lists with sequential positions (common in text indices).

### Query Performance

- Simple term queries: O(k) where k = number of matching documents
- Boolean AND: O(k1 + k2) where k1, k2 are posting list sizes
- Boolean OR: O(k1 + k2)
- Boolean NOT: O(N) where N = total documents
- Phrase queries: O(k * m) where k = docs with all terms, m = average positions per doc

## Future Enhancements

### Planned Features

1. **Database Backends**: PostgreSQL GIN, RocksDB, Redis
2. **Query Optimizations**:
   - Skipping pointers for faster AND operations
   - Early termination for top-k queries
   - Query caching
3. **Index Optimizations**:
   - Block-based compression
   - Dictionary compression for terms
   - Document pruning
4. **Performance Metrics**:
   - Latency measurement (p95, p99)
   - Throughput benchmarking
   - Memory profiling
   - Precision/recall evaluation

### Extensibility

To add a new index type:

```python
# In self_index.py
def _build_custom_index(self, files):
    # Your custom index building logic
    pass

# Update create_index method
if self.info_type == IndexInfo.CUSTOM:
    self._build_custom_index(files_list)
```

To add a new compression method:

```python
# In compression.py
class MyCompression(CompressionBase):
    def compress_posting_list(self, posting_list):
        # Your compression logic
        pass
    
    def decompress_posting_list(self, compressed_data):
        # Your decompression logic
        pass
```

## Contributing

Contributions are welcome! Please:

1. Write tests for new features
2. Follow existing code style
3. Update documentation
4. Ensure all tests pass

## License

This is an educational project for the Indexing and Retrieval course.

## References

- Assignment requirements: `Todo.md`
- Base interface: `index_base.py`
- Course materials on inverted indices and Boolean retrieval
- "Introduction to Information Retrieval" by Manning, Raghavan, and Schütze
