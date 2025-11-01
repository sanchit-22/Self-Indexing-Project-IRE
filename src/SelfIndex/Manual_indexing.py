#!/usr/bin/env python3
"""
Manual SelfIndex debugging script
Creates a specific index configuration and evaluates it with sample queries
"""

import time
import json
import statistics
import argparse
import psutil
import numpy as np
from pathlib import Path
from self_index import SelfIndex
from optimized_selfindex_evaluator import OptimizedSelfIndexEvaluator
import pandas as pd
import concurrent.futures
from collections import defaultdict

class ManualIndexDebugger:
    """Manual debugging tool for specific SelfIndex configurations"""
    
    def __init__(self):
        self.evaluator = OptimizedSelfIndexEvaluator()
        self.sample_queries = self._generate_comprehensive_query_set()
    
    def _generate_comprehensive_query_set(self):
        """Generate comprehensive query set similar to optimized evaluator"""
        query_categories = {
            'single_term': {
                'queries': ['anarchism', 'philosophy', 'politics', 'government', 'society', 'technology', 'science', 'research'],
                'purpose': 'Test basic term matching and TF-IDF scoring'
            },
            'multi_term': {
                'queries': [
                    'political philosophy movement',
                    'artificial intelligence research', 
                    'computer science technology',
                    'social economic theory',
                    'historical cultural development',
                    'scientific research methodology'
                ],
                'purpose': 'Test multi-term coordination and ranking'
            },
            'phrase_queries': {
                'queries': [
                    '"political philosophy"',
                    '"artificial intelligence"', 
                    '"social movement"',
                    '"economic theory"',
                    '"cultural development"',
                    '"scientific research"'
                ],
                'purpose': 'Test exact phrase matching and position indexing'
            },
            'long_queries': {
                'queries': [
                    'anarchism political philosophy movement skeptical authority hierarchical power structures society',
                    'artificial intelligence machine learning natural language processing computer science research development',
                    'social economic political cultural historical development theory practice implementation analysis',
                    'government authority power control society individual freedom liberty rights democracy constitutional'
                ],
                'purpose': 'Test system performance with complex queries'
            },
            'rare_terms': {
                'queries': ['epistemology', 'ontology', 'phenomenology', 'hermeneutics', 'dialectics', 'metaphysics'],
                'purpose': 'Test handling of low-frequency terms'
            },
            'common_terms': {
                'queries': ['system', 'process', 'method', 'analysis', 'development', 'research'],
                'purpose': 'Test system behavior with high-frequency terms'
            },
            'boolean_queries': {
                'queries': [
                    'anarchism AND philosophy',
                    'politics OR government', 
                    'society NOT authority',
                    '(political AND philosophy) OR (social AND movement)',
                    'technology AND (research OR development)'
                ],
                'purpose': 'Test Boolean query processing'
            },
            'empty_no_results': {
                'queries': [
                    'xyzabc123nonexistent',
                    'qqqqwwwweeeerrrr',
                    'zzzzzaaaabbbbcccc'
                ],
                'purpose': 'Test system behavior with no matches'
            }
        }
        
        # Flatten queries with metadata
        all_queries = []
        for category, info in query_categories.items():
            for query in info['queries']:
                all_queries.append({
                    'query': query,
                    'category': category,
                    'purpose': info['purpose']
                })
        
        return all_queries
    
    def run_manual_evaluation(self, doc_size, index_type, datastore, compression, query_proc, optimization):
        """Run manual evaluation for a specific configuration"""
        
        print("🔧 MANUAL SELFINDEX DEBUGGING")
        print("="*50)
        print(f"📊 Document Size: {doc_size}")
        print(f"🔧 Configuration: {index_type} | {datastore} | {compression} | {query_proc} | {optimization}")
        
        # Load data
        print("📁 Loading test data...")
        test_data = self.evaluator.load_test_data_once(doc_size)
        
        # Create index
        print("🏗️  Creating index...")
        indexer = SelfIndex(
            index_type=index_type,
            datastore=datastore,
            compression=compression,
            query_proc=query_proc,
            optimization=optimization
        )
        
        index_id = f"manual_{index_type[:4]}_{datastore[:4]}_{compression[:4]}_{query_proc[:4]}_{optimization[:4]}"
        
        start_time = time.time()
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        indexer.create_index(index_id, [(doc_id, tokens, content) for doc_id, tokens, content in test_data], pretokenized=True)
        
        index_time = time.time() - start_time
        mem_after = process.memory_info().rss / (1024 * 1024)
        memory_usage = mem_after - mem_before
        
        print(f"Index creation time: {index_time:.2f} seconds")
        print(f"Memory usage: {memory_usage:.1f} MB")
        
        # Measure metrics
        print("📏 Measuring performance metrics...")
        
        latency_metrics = self._measure_latency(indexer, self.sample_queries)
        throughput_metrics = self._measure_throughput(indexer, self.sample_queries)
        memory_metrics = self._calculate_memory_metrics(indexer, index_id, memory_usage, len(test_data))
        functional_metrics = self._measure_functional(indexer, self.sample_queries)
        
        # Display results
        self._display_results(latency_metrics, throughput_metrics, memory_metrics, functional_metrics, index_time)
        
        return {
            'config': {
                'doc_size': doc_size,
                'index_type': index_type,
                'datastore': datastore,
                'compression': compression,
                'query_proc': query_proc,
                'optimization': optimization
            },
            'metrics': {
                'latency': latency_metrics,
                'throughput': throughput_metrics,
                'memory': memory_metrics,
                'functional': functional_metrics
            },
            'index_time': index_time
        }
    
    def _measure_latency(self, indexer, query_set):
        """Measure latency for sample queries"""
        latencies = []
        category_latencies = defaultdict(list)
        
        # Warmup
        for query_info in query_set[:3]:
            try:
                indexer.query(query_info['query'])
            except:
                pass
        
        # Measure ALL queries for comprehensive latency analysis
        for query_info in query_set:
            try:
                start_time = time.time()
                result = indexer.query(query_info['query'])
                end_time = time.time()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                category_latencies[query_info['category']].append(latency_ms)
                
            except Exception:
                # Record failed query with penalty
                latencies.append(1000)
        
        if latencies:
            return {
                'mean': statistics.mean(latencies),
                'p50': np.percentile(latencies, 50),
                'p90': np.percentile(latencies, 90),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99),
                'min': min(latencies),
                'max': max(latencies),
                'total_queries': len(latencies),
                'successful_queries': len([t for t in latencies if t < 1000]),
                'category_breakdown': {cat: {
                    'mean': statistics.mean(times),
                    'p95': np.percentile(times, 95)
                } for cat, times in category_latencies.items() if times}
            }
        
        return {'error': 'No successful queries'}
    
    def _measure_throughput(self, indexer, query_set, duration=10):
        """Measure throughput"""
        # Prepare test queries for throughput measurement
        test_queries = []
        for query_info in query_set[:10]:
            test_queries.append(query_info['query'])
        
        count = 0
        start = time.time()
        
        while time.time() - start < duration:
            query = test_queries[count % len(test_queries)]
            try:
                indexer.query(query)
                count += 1
            except:
                pass
        
        elapsed = time.time() - start
        qps = count / elapsed if elapsed > 0 else 0
        
        return {
            'queries_per_second': qps,
            'total_queries': count,
            'duration_seconds': elapsed
        }
    
    def _calculate_memory_metrics(self, indexer, index_id, creation_memory, doc_count):
        """Calculate memory-related metrics"""
        index_size = self.evaluator._estimate_index_size(indexer, index_id)
        
        return {
            'index_size_mb': index_size,
            'creation_memory_mb': creation_memory,
            'docs_per_mb': doc_count / index_size if index_size > 0 else 0
        }
    
    def _measure_functional(self, indexer, query_set):
        """Measure functional metrics"""
        total_results = 0
        successful_queries = 0
        
        for query_info in query_set:
            try:
                result_json = indexer.query(query_info['query'])
                result = json.loads(result_json)
                if 'results' in result and result['results']:
                    total_results += len(result['results'])
                    successful_queries += 1
            except:
                pass
        
        return {
            'successful_queries': successful_queries,
            'total_queries': len(query_set),
            'avg_results_per_query': total_results / len(query_set) if query_set else 0
        }
    
    def _display_results(self, latency, throughput, memory, functional, index_time):
        """Display results in a nice format"""
        print("\n📊 PERFORMANCE RESULTS")
        print("="*50)
        
        print("⏱️  LATENCY:")
        print(f"   Mean: {latency['mean']:.2f} ms")
        print(f"   P95: {latency['p95']:.2f} ms")
        print(f"   P99: {latency['p99']:.2f} ms")
        print(f"   Total Queries: {latency['total_queries']}")
        print(f"   Successful: {latency['successful_queries']}")
        
        if 'category_breakdown' in latency:
            print("   Category Breakdown:")
            for cat, stats in latency['category_breakdown'].items():
                print(f"     {cat}: mean={stats['mean']:.2f}ms, p95={stats['p95']:.2f}ms")
        
        print("🚀 THROUGHPUT:")
        print(f"   Queries/sec: {throughput['queries_per_second']:.1f}")
        print(f"   Total Queries: {throughput['total_queries']}")
        
        print("💾 MEMORY:")
        print(f"   Index Size: {memory['index_size_mb']:.1f} MB")
        print(f"   Creation Memory: {memory['creation_memory_mb']:.1f} MB")
        print(f"   Docs/MB: {memory['docs_per_mb']:.1f}")
        
        print("🎯 FUNCTIONAL:")
        print(f"   Successful Queries: {functional['successful_queries']}/{functional['total_queries']}")
        print(f"   Avg Results/Query: {functional['avg_results_per_query']:.1f}")
        
        print("🏗️  INDEX CREATION:")
        print(f"   Time: {index_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Manual SelfIndex Debugging Tool')
    parser.add_argument('--doc-size', type=int, default=1000, help='Number of documents to index')
    parser.add_argument('--index-type', choices=['BOOLEAN', 'WORDCOUNT', 'TFIDF'], default='BOOLEAN', help='Index type')
    parser.add_argument('--datastore', choices=['CUSTOM', 'DB1', 'DB2'], default='CUSTOM', help='Datastore type')
    parser.add_argument('--compression', choices=['NONE', 'CODE', 'CLIB'], default='NONE', help='Compression type')
    parser.add_argument('--query-proc', choices=['TERMatat', 'DOCatat'], default='TERMatat', help='Query processing type')
    parser.add_argument('--optimization', choices=['Null', 'Skipping'], default='Null', help='Optimization type')
    
    args = parser.parse_args()
    
    debugger = ManualIndexDebugger()
    result = debugger.run_manual_evaluation(
        args.doc_size,
        args.index_type,
        args.datastore,
        args.compression,
        args.query_proc,
        args.optimization
    )
    
    # Save to file
    output_file = Path("manual_debug_results.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    main()