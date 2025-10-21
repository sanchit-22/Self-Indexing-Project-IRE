"""
Example usage and demonstration of SelfIndex.

This script demonstrates how to create and use different SelfIndex variants.
"""

from self_index import SelfIndex, create_self_index
from pathlib import Path
import json
import pandas as pd


def load_preprocessed_dataset(path="dataset/preprocessed_dataset.csv", max_docs=None):
    df = pd.read_csv(path)
    if max_docs:
        df = df.head(max_docs)
    return [(str(row['id']), str(row['text'])) for _, row in df.iterrows()]


def demo_basic_usage():
    """Demonstrate basic SelfIndex usage."""
    print("=" * 70)
    print("SelfIndex Basic Usage Demo")
    print("=" * 70)
    
    # Sample documents
    documents = load_preprocessed_dataset()
    
    # Create a Boolean index
    print("\n1. Creating Boolean Index...")
    bool_index = create_self_index(
        index_id='demo_boolean',
        files=documents,
        info='BOOLEAN',
        dstore='CUSTOM',
        qproc='TERMatat',
        compr='NONE',
        optim='Null'
    )
    print(f"   Index created: {bool_index.identifier_short}")
    print(f"   Documents indexed: {bool_index.num_docs}")
    print(f"   Vocabulary size: {len(bool_index.vocabulary)}")
    
    # Query the index
    print("\n2. Querying Boolean Index...")
    queries = [
        "quick",
        '"quick" AND "brown"',
        '"quick" OR "fast"',
        'NOT "lazy"',
    ]
    
    for query in queries:
        result = json.loads(bool_index.query(query))
        print(f"\n   Query: {query}")
        print(f"   Results: {result['num_results']} documents")
        for i, res in enumerate(result['results'][:3], 1):
            print(f"     {i}. {res['doc_id']} (score: {res['score']:.2f})")
    
    # Create a WordCount index
    print("\n3. Creating WordCount Index...")
    wc_index = create_self_index(
        index_id='demo_wordcount',
        files=documents,
        info='WORDCOUNT',
        dstore='CUSTOM'
    )
    print(f"   Index created: {wc_index.identifier_short}")
    
    # Compare ranking
    print("\n4. Comparing Boolean vs WordCount ranking...")
    test_query = "quick"
    
    bool_result = json.loads(bool_index.query(test_query))
    wc_result = json.loads(wc_index.query(test_query))
    
    print(f"\n   Query: {test_query}")
    print(f"   Boolean results: {bool_result['results']}")
    print(f"   WordCount results: {wc_result['results']}")
    
    # Create a TF-IDF index
    print("\n5. Creating TF-IDF Index...")
    tfidf_index = create_self_index(
        index_id='demo_tfidf',
        files=documents,
        info='TFIDF',
        dstore='CUSTOM'
    )
    print(f"   Index created: {tfidf_index.identifier_short}")
    
    tfidf_result = json.loads(tfidf_index.query(test_query))
    print(f"   TF-IDF results: {tfidf_result['results']}")


def demo_boolean_queries():
    """Demonstrate complex Boolean queries."""
    print("\n" + "=" * 70)
    print("Boolean Query Demo")
    print("=" * 70)
    
    # Create test documents
    documents = load_preprocessed_dataset()
    
    index = create_self_index('demo_boolean_queries', documents, info='BOOLEAN')
    
    # Test various Boolean queries
    queries = [
        '"apple" AND "banana"',
        '"apple" OR "fig"',
        'NOT "apple"',
        '("apple" OR "fig") AND "grape"',
        '"apple" AND NOT "cherry"',
        '("apple" AND "banana") OR ("elderberry" AND "fig")',
    ]
    
    print(f"\nDocuments:")
    for doc_id, content in documents:
        print(f"  {doc_id}: {content}")
    
    print(f"\nBoolean Queries:")
    for query in queries:
        result = json.loads(index.query(query))
        doc_ids = [r['doc_id'] for r in result['results']]
        print(f"\n  {query}")
        print(f"    => {doc_ids if doc_ids else 'No matches'}")


def demo_phrase_queries():
    """Demonstrate phrase queries."""
    print("\n" + "=" * 70)
    print("Phrase Query Demo")
    print("=" * 70)
    
    documents = load_preprocessed_dataset()
    
    index = create_self_index('demo_phrase', documents, info='BOOLEAN')
    
    print(f"\nDocuments:")
    for doc_id, content in documents:
        print(f"  {doc_id}: {content}")
    
    phrase_queries = [
        '"quick brown"',
        '"lazy dog"',
        '"brown fox"',
    ]
    
    print(f"\nPhrase Queries:")
    for query in phrase_queries:
        result = json.loads(index.query(query))
        doc_ids = [r['doc_id'] for r in result['results']]
        print(f"\n  {query}")
        print(f"    => {doc_ids if doc_ids else 'No matches'}")


def demo_persistence():
    """Demonstrate index persistence and loading."""
    print("\n" + "=" * 70)
    print("Index Persistence Demo")
    print("=" * 70)
    
    documents = load_preprocessed_dataset()
    
    # Create and save an index
    print("\n1. Creating and saving index...")
    index1 = create_self_index('demo_persist', documents, info='BOOLEAN')
    print(f"   Created index: {index1.identifier_short}")
    print(f"   Index saved to: {index1._get_index_path('demo_persist')}")
    
    # List all indices
    print("\n2. Listing all indices...")
    all_indices = list(SelfIndex.list_indices())
    print(f"   Found {len(all_indices)} indices:")
    for idx_id in all_indices:
        print(f"     - {idx_id}")
    
    # Load the index
    print("\n3. Loading saved index...")
    index2 = SelfIndex(info='BOOLEAN', dstore='CUSTOM')
    index_path = index2._get_index_path('demo_persist')
    index2.load_index(str(index_path))
    print(f"   Loaded index with {index2.num_docs} documents")
    print(f"   Vocabulary size: {len(index2.vocabulary)}")
    
    # Query the loaded index
    result = json.loads(index2.query("test"))
    print(f"   Query result: {result['num_results']} documents found")


def demo_index_update():
    """Demonstrate index updates."""
    print("\n" + "=" * 70)
    print("Index Update Demo")
    print("=" * 70)
    
    initial_docs = [
        ("doc1", "initial document one"),
        ("doc2", "initial document two"),
    ]
    
    print("\n1. Creating initial index...")
    index = create_self_index('demo_update', initial_docs, info='BOOLEAN')
    print(f"   Initial documents: {list(index.list_indexed_files('demo_update'))}")
    print(f"   Document count: {index.num_docs}")
    
    # Add documents
    print("\n2. Adding new documents...")
    new_docs = [
        ("doc3", "new document three"),
        ("doc4", "new document four"),
    ]
    index.update_index('demo_update', [], new_docs)
    print(f"   Updated documents: {list(index.list_indexed_files('demo_update'))}")
    print(f"   Document count: {index.num_docs}")
    
    # Remove a document
    print("\n3. Removing a document...")
    remove_docs = [("doc2", "")]
    index.update_index('demo_update', remove_docs, [])
    print(f"   Final documents: {list(index.list_indexed_files('demo_update'))}")
    print(f"   Document count: {index.num_docs}")


def demo_compression():
    """Demonstrate different compression methods."""
    print("\n" + "=" * 70)
    print("Compression Methods Demo")
    print("=" * 70)
    
    # Create a larger document set for meaningful compression
    documents = load_preprocessed_dataset()
    
    print("\n1. Creating indices with different compression methods...")
    
    # No compression
    print("\n   a) No compression (NONE):")
    idx_none = create_self_index('demo_comp_none', documents, 
                                  info='BOOLEAN', compr='NONE')
    path_none = idx_none._get_index_path('demo_comp_none')
    size_none = path_none.stat().st_size if path_none.exists() else 0
    print(f"      Index size: {size_none} bytes")
    
    # Custom compression
    print("\n   b) Custom compression (CODE):")
    idx_custom = create_self_index('demo_comp_custom', documents,
                                     info='BOOLEAN', compr='CODE')
    path_custom = idx_custom._get_index_path('demo_comp_custom')
    size_custom = path_custom.stat().st_size if path_custom.exists() else 0
    print(f"      Index size: {size_custom} bytes")
    print(f"      Ratio: {size_custom/size_none*100:.1f}%" if size_none > 0 else "")
    
    # Library compression
    print("\n   c) Library compression (CLIB):")
    idx_lib = create_self_index('demo_comp_lib', documents,
                                 info='BOOLEAN', compr='CLIB')
    path_lib = idx_lib._get_index_path('demo_comp_lib')
    size_lib = path_lib.stat().st_size if path_lib.exists() else 0
    print(f"      Index size: {size_lib} bytes")
    print(f"      Ratio: {size_lib/size_none*100:.1f}%" if size_none > 0 else "")
    
    print("\n2. Verifying all indices work correctly...")
    test_query = "word"
    
    for name, idx in [("NONE", idx_none), ("CODE", idx_custom), ("CLIB", idx_lib)]:
        result = json.loads(idx.query(test_query))
        print(f"   {name}: {result['num_results']} results")


def demo_all_variants():
    """Demonstrate all SelfIndex variants."""
    print("\n" + "=" * 70)
    print("All SelfIndex Variants")
    print("=" * 70)
    
    documents = load_preprocessed_dataset()
    
    variants = [
        ('BOOLEAN', 'CUSTOM', 'TERMatat', 'NONE', 'Null'),
        ('WORDCOUNT', 'CUSTOM', 'TERMatat', 'NONE', 'Null'),
        ('TFIDF', 'CUSTOM', 'TERMatat', 'NONE', 'Null'),
        ('BOOLEAN', 'CUSTOM', 'DOCatat', 'NONE', 'Null'),
        ('BOOLEAN', 'CUSTOM', 'TERMatat', 'CODE', 'Null'),
        ('BOOLEAN', 'CUSTOM', 'TERMatat', 'CLIB', 'Null'),
    ]
    
    print("\nCreating indices with different configurations:")
    for i, (info, dstore, qproc, compr, optim) in enumerate(variants, 1):
        idx = create_self_index(
            f'variant_{i}',
            documents,
            info=info,
            dstore=dstore,
            qproc=qproc,
            compr=compr,
            optim=optim
        )
        print(f"\n  Variant {i}: {idx.identifier_short}")
        print(f"    Long: {idx.identifier_long}")


if __name__ == '__main__':
    # Run all demos
    demo_basic_usage()
    demo_boolean_queries()
    demo_phrase_queries()
    demo_persistence()
    demo_index_update()
    
    # Note: Compression demos commented out until compression is integrated
    # demo_compression()
    
    demo_all_variants()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
