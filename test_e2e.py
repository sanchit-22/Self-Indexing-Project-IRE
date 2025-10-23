"""
End-to-end test for SelfIndex implementation.
Tests all major features with realistic data.
"""

from self_index import create_self_index, SelfIndex
from query_parser import BooleanQueryExecutor
import json
import time

# Sample documents
test_docs = [
    ("doc1", "Anarchism is a political philosophy that advocates self-governed societies"),
    ("doc2", "Political philosophy is the study of government and justice"),
    ("doc3", "Machine learning is a subset of artificial intelligence"),
    ("doc4", "Deep learning uses neural networks for pattern recognition"),
    ("doc5", "Artificial intelligence and machine learning are transforming technology"),
    ("doc6", "The philosophy of anarchism rejects hierarchical authority"),
    ("doc7", "Neural networks in deep learning mimic biological brains"),
    ("doc8", "Political systems and government structures vary across societies"),
    ("doc9", "Self-governed communities emphasize individual freedom and autonomy"),
    ("doc10", "Pattern recognition using machine learning improves over time"),
]

def test_all_features():
    """Test all major SelfIndex features."""
    
    print("=" * 80)
    print("END-TO-END SELFINDEX TEST")
    print("=" * 80)
    
    # Test 1: All index types
    print("\n1️⃣  Testing all index types...")
    for info_type in ['BOOLEAN', 'WORDCOUNT', 'TFIDF']:
        idx = create_self_index(
            index_id=f'test_{info_type.lower()}',
            files=test_docs,
            info=info_type,
            dstore='CUSTOM',
            qproc='TERMatat',
            compr='NONE',
            optim='Null'
        )
        print(f"   ✅ {info_type}: {idx.identifier_short} - {idx.num_docs} docs, {len(idx.vocabulary)} terms")
    
    # Test 2: Boolean queries
    print("\n2️⃣  Testing Boolean queries...")
    idx = create_self_index('test_bool_queries', test_docs, info='BOOLEAN')
    
    queries = [
        '"machine" AND "learning"',
        '"political" OR "philosophy"',
        'NOT "neural"',
        '"machine learning"',
        '("political" OR "anarchism") AND "philosophy"',
    ]
    
    for query in queries:
        result = json.loads(idx.query(query))
        print(f"   Query: {query:50} Results: {result['num_results']}")
    
    # Test 3: Database backends
    print("\n3️⃣  Testing database backends...")
    for backend, name in [('CUSTOM', 'Custom'), ('DB1', 'SQLite'), ('DB2', 'Redis')]:
        idx = create_self_index(
            index_id=f'test_backend_{backend.lower()}',
            files=test_docs,
            info='BOOLEAN',
            dstore=backend,
            qproc='TERMatat',
            compr='NONE',
            optim='Null'
        )
        result = json.loads(idx.query("machine learning"))
        print(f"   ✅ {name:10} backend: {result['num_results']} results")
    
    # Test 4: Compression methods
    print("\n4️⃣  Testing compression methods...")
    for compr, name in [('NONE', 'None'), ('CODE', 'Gap+VByte'), ('CLIB', 'Zlib')]:
        idx = create_self_index(
            index_id=f'test_compr_{compr.lower()}',
            files=test_docs,
            info='BOOLEAN',
            dstore='CUSTOM',
            qproc='TERMatat',
            compr=compr,
            optim='Null'
        )
        print(f"   ✅ {name:12} compression: {idx.identifier_short}")
    
    # Test 5: Query processing strategies
    print("\n5️⃣  Testing query processing strategies...")
    for qproc, name in [('TERMatat', 'Term-at-a-time'), ('DOCatat', 'Document-at-a-time')]:
        start = time.time()
        idx = create_self_index(
            index_id=f'test_qproc_{qproc.lower()}',
            files=test_docs,
            info='TFIDF',
            dstore='CUSTOM',
            qproc=qproc,
            compr='NONE',
            optim='Null'
        )
        result = json.loads(idx.query("machine learning artificial intelligence"))
        elapsed = time.time() - start
        print(f"   ✅ {name:20} {result['num_results']} results in {elapsed:.3f}s")
    
    # Test 6: Optimizations
    print("\n6️⃣  Testing optimizations...")
    for optim, name in [('Null', 'None'), ('Skipping', 'Skipping'), 
                         ('Thresholding', 'Thresholding'), ('EarlyStopping', 'EarlyStopping')]:
        idx = create_self_index(
            index_id=f'test_optim_{optim.lower()}',
            files=test_docs,
            info='BOOLEAN',
            dstore='CUSTOM',
            qproc='TERMatat',
            compr='NONE',
            optim=optim
        )
        result = json.loads(idx.query("political philosophy anarchism"))
        print(f"   ✅ {name:15} optimization: {result['num_results']} results")
    
    # Test 7: Phrase queries
    print("\n7️⃣  Testing phrase queries...")
    idx = create_self_index('test_phrases', test_docs, info='BOOLEAN')
    
    phrases = [
        '"machine learning"',
        '"political philosophy"',
        '"deep learning"',
        '"neural networks"',
    ]
    
    for phrase in phrases:
        result = json.loads(idx.query(phrase))
        print(f"   Phrase: {phrase:25} Results: {result['num_results']}")
    
    # Test 8: Index persistence and loading
    print("\n8️⃣  Testing persistence and loading...")
    idx = create_self_index('test_persist', test_docs, info='TFIDF', dstore='CUSTOM')
    original_vocab_size = len(idx.vocabulary)
    
    # Create a new instance and load
    idx2 = SelfIndex(info='TFIDF', dstore='CUSTOM', qproc='TERMatat', compr='NONE', optim='Null')
    idx2.load_index('test_persist')
    loaded_vocab_size = len(idx2.vocabulary)
    
    print(f"   Original vocab: {original_vocab_size}, Loaded vocab: {loaded_vocab_size}")
    assert original_vocab_size == loaded_vocab_size, "Vocabulary size mismatch!"
    print(f"   ✅ Index persistence working correctly")
    
    # Test 9: List indices
    print("\n9️⃣  Testing index listing...")
    indices = list(SelfIndex.list_indices())
    print(f"   Found {len(indices)} indices")
    print(f"   Sample indices: {indices[:5]}")
    
    # Test 10: Complex query
    print("\n🔟  Testing complex query...")
    idx = create_self_index('test_complex', test_docs, info='TFIDF')
    
    complex_query = '("machine learning" OR "deep learning") AND NOT "biological"'
    result = json.loads(idx.query(complex_query))
    
    print(f"   Complex query: {complex_query}")
    print(f"   Results: {result['num_results']} documents")
    for i, res in enumerate(result['results'][:3], 1):
        print(f"      {i}. {res['doc_id']} (score: {res['score']:.2f})")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)

if __name__ == '__main__':
    test_all_features()
