"""
Performance evaluation and benchmarking for SelfIndex.

This script measures and compares:
- A: Latency (response time with p95 and p99 percentiles)
- B: Throughput (queries per second)
- C: Memory footprint
- D: Functional metrics (if applicable)
"""

import time
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import statistics

from self_index import SelfIndex, create_self_index


class PerformanceEvaluator:
    """Evaluate performance of SelfIndex."""
    
    def __init__(self):
        self.results = []
    
    def generate_test_documents(self, num_docs: int = 1000) -> List[tuple]:
        """Generate test documents."""
        documents = []
        
        # Generate documents with varying content
        terms = [
            "search", "information", "retrieval", "index", "query",
            "document", "term", "frequency", "ranking", "boolean",
            "database", "system", "algorithm", "data", "structure",
            "processing", "analysis", "mining", "learning", "machine"
        ]
        
        for i in range(num_docs):
            # Create document with random terms
            import random
            random.seed(i)  # For reproducibility
            doc_terms = random.sample(terms, k=random.randint(5, 15))
            content = " ".join(doc_terms * random.randint(1, 3))
            documents.append((f"doc{i}", content))
        
        return documents
    
    def generate_test_queries(self, num_queries: int = 100) -> List[str]:
        """Generate test queries."""
        queries = []
        
        # Simple term queries
        terms = ["search", "information", "query", "document", "system"]
        for term in terms:
            queries.append(f'"{term}"')
        
        # Boolean AND queries
        queries.extend([
            '"search" AND "information"',
            '"query" AND "document"',
            '"database" AND "system"',
            '"data" AND "structure"',
            '"machine" AND "learning"',
        ])
        
        # Boolean OR queries
        queries.extend([
            '"search" OR "query"',
            '"document" OR "information"',
            '"algorithm" OR "structure"',
        ])
        
        # NOT queries
        queries.extend([
            'NOT "search"',
            'NOT "document"',
        ])
        
        # Complex queries
        queries.extend([
            '("search" OR "query") AND "information"',
            '"database" AND NOT "learning"',
            '("machine" AND "learning") OR ("data" AND "mining")',
        ])
        
        # Extend to reach num_queries
        while len(queries) < num_queries:
            queries.extend(queries[:min(10, num_queries - len(queries))])
        
        return queries[:num_queries]
    
    def measure_latency(self, index: SelfIndex, queries: List[str]) -> Dict[str, float]:
        """
        Measure query latency.
        
        Returns:
            Dictionary with mean, p95, p99 latencies in milliseconds
        """
        latencies = []
        
        for query in queries:
            start_time = time.perf_counter()
            result = index.query(query)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        latencies.sort()
        
        return {
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'p95': latencies[int(len(latencies) * 0.95)],
            'p99': latencies[int(len(latencies) * 0.99)],
            'min': min(latencies),
            'max': max(latencies),
        }
    
    def measure_throughput(self, index: SelfIndex, queries: List[str], 
                          duration_seconds: float = 5.0) -> float:
        """
        Measure query throughput.
        
        Returns:
            Queries per second
        """
        query_count = 0
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        query_idx = 0
        while time.perf_counter() < end_time:
            query = queries[query_idx % len(queries)]
            result = index.query(query)
            query_count += 1
            query_idx += 1
        
        elapsed = time.perf_counter() - start_time
        return query_count / elapsed
    
    def measure_index_size(self, index_id: str) -> Dict[str, int]:
        """
        Measure index size on disk.
        
        Returns:
            Dictionary with size in bytes
        """
        storage_dir = SelfIndex.INDEX_STORAGE_DIR
        total_size = 0
        
        # Find all files for this index
        for path in storage_dir.glob(f"{index_id}*"):
            if path.is_file():
                total_size += path.stat().st_size
        
        return {
            'total_bytes': total_size,
            'total_kb': total_size / 1024,
            'total_mb': total_size / (1024 * 1024),
        }
    
    def measure_index_creation_time(self, index_id: str, files: List[tuple],
                                   **kwargs) -> float:
        """
        Measure time to create an index.
        
        Returns:
            Time in seconds
        """
        start_time = time.perf_counter()
        index = create_self_index(index_id, files, **kwargs)
        end_time = time.perf_counter()
        
        return end_time - start_time
    
    def evaluate_variant(self, variant_name: str, documents: List[tuple],
                        queries: List[str], **config) -> Dict[str, Any]:
        """
        Evaluate a specific SelfIndex variant.
        
        Returns:
            Dictionary with performance metrics
        """
        print(f"\n{'='*70}")
        print(f"Evaluating: {variant_name}")
        print(f"Configuration: {config}")
        print(f"{'='*70}")
        
        index_id = f"perf_eval_{variant_name}"
        
        # Measure index creation time
        print("\n1. Measuring index creation time...")
        creation_time = self.measure_index_creation_time(index_id, documents, **config)
        print(f"   Creation time: {creation_time:.3f} seconds")
        
        # Load the index
        index = SelfIndex(**config)
        index_path = index._get_index_path(index_id)
        index.load_index(str(index_path))
        
        # Measure index size
        print("\n2. Measuring index size...")
        size_info = self.measure_index_size(index_id)
        print(f"   Size: {size_info['total_kb']:.2f} KB ({size_info['total_bytes']} bytes)")
        
        # Measure latency
        print("\n3. Measuring query latency...")
        latency_info = self.measure_latency(index, queries)
        print(f"   Mean latency: {latency_info['mean']:.3f} ms")
        print(f"   Median latency: {latency_info['median']:.3f} ms")
        print(f"   P95 latency: {latency_info['p95']:.3f} ms")
        print(f"   P99 latency: {latency_info['p99']:.3f} ms")
        
        # Measure throughput
        print("\n4. Measuring throughput...")
        throughput = self.measure_throughput(index, queries, duration_seconds=3.0)
        print(f"   Throughput: {throughput:.2f} queries/second")
        
        # Clean up
        index.delete_index(index_id)
        
        result = {
            'variant': variant_name,
            'config': config,
            'creation_time_seconds': creation_time,
            'index_size': size_info,
            'latency': latency_info,
            'throughput_qps': throughput,
        }
        
        self.results.append(result)
        return result
    
    def compare_variants(self, documents: List[tuple], queries: List[str]):
        """Compare different SelfIndex variants."""
        
        variants = [
            ("Boolean_NoCompr", {
                'info': 'BOOLEAN',
                'dstore': 'CUSTOM',
                'qproc': 'TERMatat',
                'compr': 'NONE',
                'optim': 'Null'
            }),
            ("Boolean_CustomCompr", {
                'info': 'BOOLEAN',
                'dstore': 'CUSTOM',
                'qproc': 'TERMatat',
                'compr': 'CODE',
                'optim': 'Null'
            }),
            ("Boolean_LibCompr", {
                'info': 'BOOLEAN',
                'dstore': 'CUSTOM',
                'qproc': 'TERMatat',
                'compr': 'CLIB',
                'optim': 'Null'
            }),
            ("WordCount_NoCompr", {
                'info': 'WORDCOUNT',
                'dstore': 'CUSTOM',
                'qproc': 'TERMatat',
                'compr': 'NONE',
                'optim': 'Null'
            }),
            ("TFIDF_NoCompr", {
                'info': 'TFIDF',
                'dstore': 'CUSTOM',
                'qproc': 'TERMatat',
                'compr': 'NONE',
                'optim': 'Null'
            }),
        ]
        
        for variant_name, config in variants:
            try:
                self.evaluate_variant(variant_name, documents, queries, **config)
            except Exception as e:
                print(f"Error evaluating {variant_name}: {e}")
                import traceback
                traceback.print_exc()
    
    def print_comparison_table(self):
        """Print comparison table of all results."""
        if not self.results:
            print("No results to compare")
            return
        
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*70)
        
        # Header
        print(f"\n{'Variant':<25} {'Creation(s)':<12} {'Size(KB)':<12} "
              f"{'P95(ms)':<10} {'Throughput':<12}")
        print("-" * 70)
        
        # Rows
        for result in self.results:
            variant = result['variant']
            creation = result['creation_time_seconds']
            size_kb = result['index_size']['total_kb']
            p95 = result['latency']['p95']
            throughput = result['throughput_qps']
            
            print(f"{variant:<25} {creation:<12.3f} {size_kb:<12.2f} "
                  f"{p95:<10.3f} {throughput:<12.2f}")
        
        print("\n" + "="*70)
    
    def save_results(self, filename: str = "performance_results.json"):
        """Save results to JSON file."""
        output_path = Path("results") / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


def main():
    """Run performance evaluation."""
    print("SelfIndex Performance Evaluation")
    print("="*70)
    
    evaluator = PerformanceEvaluator()
    
    # Generate test data
    print("\nGenerating test data...")
    num_docs = 1000 if len(sys.argv) < 2 else int(sys.argv[1])
    num_queries = 100 if len(sys.argv) < 3 else int(sys.argv[2])
    
    documents = evaluator.generate_test_documents(num_docs)
    queries = evaluator.generate_test_queries(num_queries)
    
    print(f"Generated {len(documents)} documents")
    print(f"Generated {len(queries)} queries")
    
    # Run comparisons
    evaluator.compare_variants(documents, queries)
    
    # Print summary
    evaluator.print_comparison_table()
    
    # Save results
    evaluator.save_results(f"selfindex_perf_{num_docs}docs.json")


if __name__ == '__main__':
    main()
