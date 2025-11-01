#!/usr/bin/env python3
"""
Test script to compare Term-at-a-time vs Document-at-a-time query processing performance
"""

import time
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from self_index import SelfIndex
from optimized_selfindex_evaluator import OptimizedSelfIndexEvaluator

def test_query_algorithms():
    """Test both query processing algorithms with same data and queries"""
    
    print("🧪 TESTING QUERY ALGORITHM PERFORMANCE")
    print("=" * 50)
    
    # Load small dataset for testing
    evaluator = OptimizedSelfIndexEvaluator()
    test_data = evaluator.load_test_data_once(500)  # Small dataset for quick testing
    
    # Sample queries for testing
    test_queries = [
        'anarchism',
        'political philosophy',
        'government authority',
        'anarchism AND philosophy',
        'technology research development',
        '"artificial intelligence"',
        'society OR government',
        'political philosophy movement'
    ]
    
    print(f"📊 Test Setup:")
    print(f"   Documents: {len(test_data)}")
    print(f"   Test Queries: {len(test_queries)}")
    print()
    
    # Test Term-at-a-time
    print("🔍 Testing TERM-AT-A-TIME algorithm...")
    indexer_term = SelfIndex(
        index_type='BOOLEAN',
        datastore='CUSTOM', 
        compression='NONE',
        query_proc='TERMatat',
        optimization='Null'
    )
    
    indexer_term.create_index('test_term', test_data, pretokenized=True)
    
    term_latencies = []
    term_results_count = []
    
    # Warmup
    for _ in range(3):
        indexer_term.query('warmup')
    
    start_time = time.time()
    for query in test_queries:
        query_start = time.time()
        results = indexer_term.query(query)
        query_end = time.time()
        
        latency = (query_end - query_start) * 1000  # Convert to ms
        term_latencies.append(latency)
        
        if isinstance(results, str):
            import json
            results = json.loads(results)
        term_results_count.append(len(results.get('results', [])))
    
    term_total_time = time.time() - start_time
    
    print(f"   ✅ Term-at-a-time completed")
    print(f"   Average latency: {sum(term_latencies)/len(term_latencies):.2f} ms")
    print(f"   Total time: {term_total_time:.3f} seconds")
    
    # Test Document-at-a-time
    print("\n🔍 Testing DOCUMENT-AT-A-TIME algorithm...")
    indexer_doc = SelfIndex(
        index_type='BOOLEAN',
        datastore='CUSTOM',
        compression='NONE', 
        query_proc='DOCatat',
        optimization='Null'
    )
    
    indexer_doc.create_index('test_doc', test_data, pretokenized=True)
    
    doc_latencies = []
    doc_results_count = []
    
    # Warmup
    for _ in range(3):
        indexer_doc.query('warmup')
    
    start_time = time.time()
    for query in test_queries:
        query_start = time.time()
        results = indexer_doc.query(query)
        query_end = time.time()
        
        latency = (query_end - query_start) * 1000  # Convert to ms
        doc_latencies.append(latency)
        
        if isinstance(results, str):
            import json
            results = json.loads(results)
        doc_results_count.append(len(results.get('results', [])))
    
    doc_total_time = time.time() - start_time
    
    print(f"   ✅ Document-at-a-time completed")
    print(f"   Average latency: {sum(doc_latencies)/len(doc_latencies):.2f} ms")
    print(f"   Total time: {doc_total_time:.3f} seconds")
    
    # Compare results
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"   Algorithm           | Avg Latency | Total Time | Speedup")
    print(f"   ------------------- | ----------- | ---------- | -------")
    print(f"   Term-at-a-time      | {sum(term_latencies)/len(term_latencies):8.2f} ms | {term_total_time:7.3f} s |   1.00x")
    print(f"   Document-at-a-time  | {sum(doc_latencies)/len(doc_latencies):8.2f} ms | {doc_total_time:7.3f} s | {term_total_time/doc_total_time:6.2f}x")
    
    # Verify both algorithms return same results
    print(f"\n🔍 CORRECTNESS CHECK:")
    results_match = True
    for i in range(len(test_queries)):
        if term_results_count[i] != doc_results_count[i]:
            results_match = False
            print(f"   ❌ Query '{test_queries[i]}': Term={term_results_count[i]} vs Doc={doc_results_count[i]} results")
        else:
            print(f"   ✅ Query '{test_queries[i]}': Both algorithms returned {term_results_count[i]} results")
    
    if results_match:
        print(f"   🎉 All queries returned identical result counts!")
    else:
        print(f"   ⚠️  Some queries returned different result counts - may indicate a bug")
    
    # Detailed latency analysis
    print(f"\n📈 DETAILED LATENCY ANALYSIS:")
    print(f"   Query                         | Term (ms) | Doc (ms) | Ratio")
    print(f"   ----------------------------- | --------- | -------- | -----")
    for i, query in enumerate(test_queries):
        ratio = doc_latencies[i] / term_latencies[i] if term_latencies[i] > 0 else 1.0
        print(f"   {query:29} | {term_latencies[i]:8.2f} | {doc_latencies[i]:7.2f} | {ratio:4.2f}x")

if __name__ == "__main__":
    test_query_algorithms()