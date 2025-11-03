#!/usr/bin/env python3
"""
Manual Test Script for Single SelfIndex Configuration
Allows testing any specific configuration with the same queries used in full evaluation
"""

import sys
import json
import time
from pathlib import Path
from self_index import SelfIndex
import pandas as pd
from tqdm import tqdm

class ManualIndexTester:
    """Test a single index configuration with comprehensive metrics"""
    
    def __init__(self):
        self.test_data = None
        self.query_set = None
        
    def load_test_data(self, max_docs=50000):
        """Load preprocessed dataset"""
        print(f"📁 Loading preprocessed dataset...")
        csv_path = Path("../Dataset/PreProcessedData/preprocessed_dataset.csv")
        
        if not csv_path.exists():
            print(f"❌ ERROR: CSV not found at {csv_path}")
            sys.exit(1)
            
        df = pd.read_csv(csv_path)
        if max_docs:
            df = df.head(max_docs)
        
        def parse_tokens(tok_str):
            """Parse tokens from space-separated string format (as saved in CSV)"""
            try:
                if isinstance(tok_str, str) and tok_str.strip():
                    # Split on whitespace to get list of tokens
                    tokens = tok_str.strip().split()
                    return [token for token in tokens if token]  # Remove empty tokens
                return []
            except Exception:
                return []
        
        documents = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading docs"):
            doc_id = str(row['id'])
            tokens = parse_tokens(row['processed_tokens'])
            content = row['original_text']
            title = row.get('title', doc_id)
            documents.append((doc_id, tokens, content))
        
        self.test_data = documents
        print(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def generate_query_set(self):
        """Generate the same comprehensive query set used in full evaluation"""
        print(f"🧠 Generating query set...")
        
        query_categories = {
            'single_term': ['anarchism', 'philosophy', 'politics', 'government', 'society', 'technology', 'science', 'research'],
            'multi_term': [
                'political philosophy movement',
                'artificial intelligence research', 
                'computer science technology',
                'social economic theory',
                'historical cultural development',
                'scientific research methodology'
            ],
            'phrase_queries': [
                '"political philosophy"',
                '"artificial intelligence"', 
                '"social movement"',
                '"economic theory"',
                '"cultural development"',
                '"scientific research"'
            ],
            'long_queries': [
                'anarchism political philosophy movement skeptical authority hierarchical power structures society',
                'artificial intelligence machine learning natural language processing computer science research development',
                'social economic political cultural historical development theory practice implementation analysis',
                'government authority power control society individual freedom liberty rights democracy constitutional'
            ],
            'rare_terms': ['epistemology', 'ontology', 'phenomenology', 'hermeneutics', 'dialectics', 'metaphysics'],
            'common_terms': ['system', 'process', 'method', 'analysis', 'development', 'research'],
            'boolean_queries': [
                'anarchism AND philosophy',
                'politics OR government', 
                'society NOT authority',
                '(political AND philosophy) OR (social AND movement)',
                'technology AND (research OR development)'
            ],
            'empty_no_results': ['xyzabc123nonexistent', 'qqqqwwwweeeerrrr', 'zzzzzaaaabbbbcccc']
        }
        
        all_queries = []
        for category, queries in query_categories.items():
            for q in queries:
                all_queries.append({'query': q, 'category': category})
        
        self.query_set = all_queries
        print(f"✅ Generated {len(all_queries)} queries across {len(query_categories)} categories")
        return all_queries
    
    def test_configuration(self, index_type='BOOLEAN', datastore='CUSTOM', compression='NONE', 
                           query_proc='TERMatat', optimization='Null', sample_size=50000):
        """Test a single configuration with all metrics"""
        
        print(f"\n{'='*80}")
        print(f"TESTING CONFIGURATION")
        print(f"{'='*80}")
        print(f"  Index Type:    {index_type}")
        print(f"  Datastore:     {datastore}")
        print(f"  Compression:   {compression}")
        print(f"  Query Proc:    {query_proc}")
        print(f"  Optimization:  {optimization}")
        print(f"  Sample Size:   {sample_size} documents")
        print(f"{'='*80}\n")
        
        # Load data if not already loaded
        if self.test_data is None:
            self.load_test_data(sample_size)
        if self.query_set is None:
            self.generate_query_set()
        
        # Create index
        indexer = SelfIndex(
            index_type=index_type,
            datastore=datastore,
            compression=compression,
            query_proc=query_proc,
            optimization=optimization
        )
        
        index_id = f"manual_test_{index_type}_{datastore}_{compression}_{query_proc}_{optimization}"
        
        print(f"📊 Step 1: Creating Index")
        print(f"   Index ID: {index_id}")
        
        start_time = time.time()
        indexer.create_index(index_id, self.test_data, pretokenized=True)
        creation_time = time.time() - start_time
        
        print(f"   ✅ Index created in {creation_time:.2f}s")
        
        # Measure latency
        print(f"\n📊 Step 2: Measuring Latency")
        latency_results = self._measure_latency(indexer, self.query_set)
        
        # Measure throughput
        print(f"\n📊 Step 3: Measuring Throughput")
        throughput_results = self._measure_throughput(indexer, self.query_set)
        
        # Estimate memory
        print(f"\n📊 Step 4: Estimating Memory Usage")
        memory_results = self._estimate_memory(indexer, index_id)
        
        # Print summary
        self._print_summary(index_type, datastore, compression, query_proc, optimization,
                           creation_time, latency_results, throughput_results, memory_results)
        
        return {
            'config': {
                'index_type': index_type,
                'datastore': datastore,
                'compression': compression,
                'query_proc': query_proc,
                'optimization': optimization
            },
            'creation_time': creation_time,
            'latency': latency_results,
            'throughput': throughput_results,
            'memory': memory_results
        }
    
    def _measure_latency(self, indexer, query_set):
        """Measure query latency with percentiles"""
        latencies = []
        
        print(f"   Running {len(query_set)} queries...")
        for query_item in tqdm(query_set, desc="Latency test"):
            query = query_item['query']
            
            start = time.perf_counter()
            result = indexer.query(query)
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        # Calculate percentiles
        import statistics
        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 20 else latencies[-1]
        p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) > 100 else latencies[-1]
        mean = statistics.mean(latencies)
        
        results = {
            'p50': round(p50, 2),
            'p95': round(p95, 2),
            'p99': round(p99, 2),
            'mean': round(mean, 2),
            'min': round(min(latencies), 2),
            'max': round(max(latencies), 2)
        }
        
        print(f"   ✅ Latency results:")
        print(f"      Mean: {results['mean']}ms")
        print(f"      P50:  {results['p50']}ms")
        print(f"      P95:  {results['p95']}ms")
        print(f"      P99:  {results['p99']}ms")
        
        return results
    
    def _measure_throughput(self, indexer, query_set, duration_seconds=15):
        """Measure queries per second"""
        print(f"   Running queries for {duration_seconds} seconds...")
        
        query_count = 0
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        query_idx = 0
        while time.time() < end_time:
            query = query_set[query_idx % len(query_set)]['query']
            indexer.query(query)
            query_count += 1
            query_idx += 1
        
        elapsed = time.time() - start_time
        qps = query_count / elapsed
        
        results = {
            'qps': round(qps, 1),
            'queries_executed': query_count,
            'duration': round(elapsed, 2)
        }
        
        print(f"   ✅ Throughput: {results['qps']} qps")
        print(f"      ({query_count} queries in {elapsed:.2f}s)")
        
        return results
    
    def _estimate_memory(self, indexer, index_id):
        """Estimate index memory usage"""
        index_dir = indexer.data_dir / index_id
        
        if index_dir.exists():
            total_size = sum(f.stat().st_size for f in index_dir.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
        else:
            size_mb = 0
        
        results = {
            'index_size_mb': round(size_mb, 2),
            'index_path': str(index_dir)
        }
        
        print(f"   ✅ Memory: {results['index_size_mb']} MB")
        print(f"      Path: {results['index_path']}")
        
        return results
    
    def _print_summary(self, index_type, datastore, compression, query_proc, optimization,
                       creation_time, latency, throughput, memory):
        """Print comprehensive summary"""
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Configuration: {index_type}_{datastore}_{compression}_{query_proc}_{optimization}")
        print(f"\n📊 Performance Metrics:")
        print(f"   Index Creation:  {creation_time:.2f}s")
        print(f"   Latency (mean):  {latency['mean']}ms")
        print(f"   Latency (P95):   {latency['p95']}ms")
        print(f"   Throughput:      {throughput['qps']} qps")
        print(f"   Memory:          {memory['index_size_mb']} MB")
        print(f"{'='*80}\n")

def main():
    """Main function with command-line interface"""
    
    print("🧪 MANUAL SELFINDEX CONFIGURATION TESTER")
    print("="*80)
    
    # Parse command-line arguments
    if len(sys.argv) < 6:
        print("\nUsage: python manual_test_index.py <index_type> <datastore> <compression> <query_proc> <optimization> [sample_size]")
        print("\nParameters:")
        print("  index_type:   BOOLEAN | WORDCOUNT | TFIDF")
        print("  datastore:    CUSTOM | DB1 | DB2")
        print("  compression:  NONE | CODE | CLIB")
        print("  query_proc:   TERMatat | DOCatat")
        print("  optimization: Null | Skipping")
        print("  sample_size:  Number of documents (default: 50000)")
        print("\nExamples:")
        print("  # Test Boolean index with no compression")
        print("  python manual_test_index.py BOOLEAN CUSTOM NONE TERMatat Null 10000")
        print("\n  # Test TFIDF with CODE compression and skip pointers")
        print("  python manual_test_index.py TFIDF DB1 CODE DOCatat Skipping 50000")
        print("\n  # Test the problematic configuration")
        print("  python manual_test_index.py BOOLEAN CUSTOM NONE DOCatat Skipping 50000")
        print()
        sys.exit(1)
    
    index_type = sys.argv[1]
    datastore = sys.argv[2]
    compression = sys.argv[3]
    query_proc = sys.argv[4]
    optimization = sys.argv[5]
    sample_size = int(sys.argv[6]) if len(sys.argv) > 6 else 50000
    
    # Validate inputs
    valid_index_types = ['BOOLEAN', 'WORDCOUNT', 'TFIDF']
    valid_datastores = ['CUSTOM', 'DB1', 'DB2']
    valid_compressions = ['NONE', 'CODE', 'CLIB']
    valid_query_procs = ['TERMatat', 'DOCatat']
    valid_optimizations = ['Null', 'Skipping']
    
    if index_type not in valid_index_types:
        print(f"❌ Invalid index_type: {index_type}. Must be one of {valid_index_types}")
        sys.exit(1)
    if datastore not in valid_datastores:
        print(f"❌ Invalid datastore: {datastore}. Must be one of {valid_datastores}")
        sys.exit(1)
    if compression not in valid_compressions:
        print(f"❌ Invalid compression: {compression}. Must be one of {valid_compressions}")
        sys.exit(1)
    if query_proc not in valid_query_procs:
        print(f"❌ Invalid query_proc: {query_proc}. Must be one of {valid_query_procs}")
        sys.exit(1)
    if optimization not in valid_optimizations:
        print(f"❌ Invalid optimization: {optimization}. Must be one of {valid_optimizations}")
        sys.exit(1)
    
    # Run test
    tester = ManualIndexTester()
    results = tester.test_configuration(
        index_type=index_type,
        datastore=datastore,
        compression=compression,
        query_proc=query_proc,
        optimization=optimization,
        sample_size=sample_size
    )
    
    # Save results to JSON
    output_file = f"manual_test_results_{index_type}_{datastore}_{compression}_{query_proc}_{optimization}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Results saved to: {output_file}")
    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()
