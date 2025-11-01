#!/usr/bin/env python3
"""
Complete evaluation framework for all 108 SelfIndex variants
Compares against ESIndex-v1.0 baseline
"""

import time
import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
from self_index import SelfIndex
from datasets import load_dataset
from tqdm import tqdm
import psutil
import itertools
from datetime import datetime
import pandas as pd
import seaborn as sns

class SelfIndexEvaluator:
    """Comprehensive evaluation framework for all 108 SelfIndex variant combinations"""
    
    def __init__(self):
        self.results = {}
        self.test_data = None
        self.query_set = None
        self.esindex_baseline = {
            'latency': {'p95': 18.88, 'p99': 26.14, 'mean': 12.57},
            'throughput': {'single_thread_qps': 91.63, 'multi_thread_qps': 375.91},
            'memory': {'index_size_mb': 140.90, 'documents_per_mb': 354.9},
            'functional': {'mean_average_precision': 0.050, 'coverage_rate': 1.000}
        }
        
    def load_test_data(self, max_docs=500):
        """Load test dataset"""
        print(f"📁 Loading test data ({max_docs} documents)...")
        
        try:
            # Use local cached data
            cache_path = Path("local_wikipedia_data")
            if cache_path.exists():
                ds = load_dataset("wikimedia/wikipedia", "20231101.en", cache_dir=cache_path, split="train")
            else:
                ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
            
            documents = []
            
            for idx, item in enumerate(tqdm(ds, desc="Loading documents", total=max_docs)):
                if idx >= max_docs:
                    break
                
                doc_id = str(item['id'])
                content = item['text']
                documents.append((doc_id, content))
            
            self.test_data = documents
            print(f"✅ Loaded {len(documents)} documents")
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            # Create sample data as fallback
            self.test_data = [
                (f"doc_{i}", f"Sample document {i} content with political philosophy anarchism government society technology artificial intelligence machine learning research science")
                for i in range(max_docs)
            ]
            print(f"✅ Created {len(self.test_data)} sample documents")
    
    def generate_query_set(self):
        """Generate diverse query set for testing"""
        self.query_set = [
            "anarchism",
            "political philosophy", 
            "government society",
            "technology research",
            "artificial intelligence",
            "computer science",
            "social movement",
            "economic theory",
            "machine learning",
            "natural language"
        ]
    
    def generate_all_configurations(self):
        """Generate all 108 possible SelfIndex configurations"""
        
        index_types = ['BOOLEAN', 'WORDCOUNT', 'TFIDF']  # x=1,2,3
        datastores = ['CUSTOM', 'DB1', 'DB2']             # y=1,2,3  
        compressions = ['NONE', 'CODE', 'CLIB']           # z=1,2,3
        optimizations = ['Null', 'Skipping']              # i=0,1
        query_procs = ['TERMatat', 'DOCatat']             # q=T,D
        
        configurations = []
        
        print(f"🔧 Generating all possible configurations...")
        print(f"   Index types: {len(index_types)} options")
        print(f"   Datastores: {len(datastores)} options") 
        print(f"   Compressions: {len(compressions)} options")
        print(f"   Optimizations: {len(optimizations)} options")
        print(f"   Query processing: {len(query_procs)} options")
        
        total_combinations = len(index_types) * len(datastores) * len(compressions) * len(optimizations) * len(query_procs)
        print(f"   📊 Total combinations: {total_combinations}")
        
        config_id = 0
        for index_type, datastore, compression, optimization, query_proc in itertools.product(
            index_types, datastores, compressions, optimizations, query_procs
        ):
            config_id += 1
            
            name = f"SelfIndex_{config_id:03d}_{index_type[:4]}_{datastore[:4]}_{compression[:4]}_{optimization[:4]}_{query_proc[:4]}"
            
            configurations.append({
                'name': name,
                'config_id': config_id,
                'index_type': index_type,
                'datastore': datastore,
                'compression': compression,
                'query_proc': query_proc,
                'optimization': optimization,
                'x': index_types.index(index_type) + 1,
                'y': datastores.index(datastore) + 1,
                'z': compressions.index(compression) + 1,
                'i': optimizations.index(optimization),
                'q': 'T' if query_proc == 'TERMatat' else 'D'
            })
        
        print(f"✅ Generated {len(configurations)} configurations")
        return configurations
    
    def run_evaluation(self, max_configs=20, sample_size=100):
        """Run evaluation of selected SelfIndex variants"""
        
        if not self.test_data:
            self.load_test_data(sample_size)
        
        if not self.query_set:
            self.generate_query_set()
        
        all_configurations = self.generate_all_configurations()
        
        # Select representative configurations for testing
        if max_configs and max_configs < len(all_configurations):
            # Select diverse configurations
            selected_configs = []
            
            # Always include basic configurations
            basic_configs = [
                ('BOOLEAN', 'CUSTOM', 'NONE', 'Null', 'TERMatat'),
                ('WORDCOUNT', 'CUSTOM', 'NONE', 'Null', 'TERMatat'),
                ('TFIDF', 'CUSTOM', 'NONE', 'Null', 'TERMatat'),
                ('BOOLEAN', 'DB1', 'NONE', 'Null', 'TERMatat'),
                ('BOOLEAN', 'DB2', 'NONE', 'Null', 'TERMatat'),
                ('BOOLEAN', 'CUSTOM', 'CODE', 'Null', 'TERMatat'),
                ('BOOLEAN', 'CUSTOM', 'CLIB', 'Null', 'TERMatat'),
                ('BOOLEAN', 'CUSTOM', 'NONE', 'Skipping', 'TERMatat'),
                ('BOOLEAN', 'CUSTOM', 'NONE', 'Null', 'DOCatat'),
                ('TFIDF', 'CUSTOM', 'CLIB', 'Skipping', 'DOCatat'),
            ]
            
            for target in basic_configs:
                for config in all_configurations:
                    if (config['index_type'], config['datastore'], config['compression'], 
                        config['optimization'], config['query_proc']) == target:
                        selected_configs.append(config)
                        break
            
            # Add random additional configs to reach max_configs
            remaining = [c for c in all_configurations if c not in selected_configs]
            import random
            random.shuffle(remaining)
            selected_configs.extend(remaining[:max_configs - len(selected_configs)])
            
            configurations = selected_configs[:max_configs]
            print(f"⚠️  Testing {len(configurations)} representative configurations")
        else:
            configurations = all_configurations
        
        print(f"\n🚀 COMPREHENSIVE SELFINDEX EVALUATION")
        print(f"{'='*80}")
        print(f"Testing {len(configurations)} configurations...")
        print(f"Sample size: {len(self.test_data)} documents")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        successful_configs = 0
        failed_configs = 0
        
        for i, config in enumerate(configurations, 1):
            print(f"\n📊 [{i}/{len(configurations)}] Testing: {config['name']}")
            
            try:
                result = self._evaluate_single_configuration(config)
                self.results[config['name']] = result
                
                if 'error' not in result:
                    successful_configs += 1
                    print(f"    ✅ Success - Index: {result.get('index_time', 0):.2f}s, Query: {result.get('avg_query_time', 0):.2f}ms")
                else:
                    failed_configs += 1
                    print(f"    ❌ Failed: {result['error']}")
                
            except Exception as e:
                failed_configs += 1
                error_msg = str(e)
                print(f"    ❌ Exception: {error_msg}")
                self.results[config['name']] = {'error': error_msg, 'config': config}
        
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"   Total configurations: {len(configurations)}")
        print(f"   Successful: {successful_configs}")
        print(f"   Failed: {failed_configs}")
        print(f"   Success rate: {successful_configs/len(configurations)*100:.1f}%")
        
        self._generate_all_plots()
        self._generate_comparison_report()
        
        return self.results
    
    def _evaluate_single_configuration(self, config):
        """Evaluate a single SelfIndex configuration"""
        
        indexer = SelfIndex(
            index_type=config['index_type'],
            datastore=config['datastore'],
            compression=config['compression'],
            query_proc=config['query_proc'],
            optimization=config['optimization']
        )
        
        result = {'config': config}
        
        try:
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 * 1024)
            
            index_id = f"eval_{config['config_id']}"
            start_time = time.time()
            indexer.create_index(index_id, self.test_data)
            index_time = time.time() - start_time
            
            mem_after = process.memory_info().rss / (1024 * 1024)
            memory_usage = mem_after - mem_before
            
            query_times = []
            successful_queries = 0
            
            for query in self.query_set:
                try:
                    start_time = time.time()
                    query_result = indexer.query(query)
                    query_time = (time.time() - start_time) * 1000
                    
                    parsed_result = json.loads(query_result)
                    if 'error' not in parsed_result:
                        query_times.append(query_time)
                        successful_queries += 1
                        
                except Exception:
                    pass
            
            if query_times:
                avg_query_time = statistics.mean(query_times)
                p95_query_time = np.percentile(query_times, 95)
                throughput = 1000 / avg_query_time if avg_query_time > 0 else 0
            else:
                avg_query_time = float('inf')
                p95_query_time = float('inf')
                throughput = 0
            
            index_size = self._estimate_index_size(indexer, index_id)
            
            indexer.delete_index(index_id)
            
            result.update({
                'index_time': index_time,
                'memory_usage': memory_usage,
                'index_size_mb': index_size,
                'avg_query_time': avg_query_time,
                'p95_query_time': p95_query_time,
                'throughput_qps': throughput,
                'successful_queries': successful_queries,
                'total_queries': len(self.query_set),
                'query_success_rate': successful_queries / len(self.query_set) if self.query_set else 0
            })
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _estimate_index_size(self, indexer, index_id):
        """Estimate index size on disk"""
        try:
            if indexer.datastore == 'CUSTOM':
                index_dir = indexer.data_dir / index_id
                if index_dir.exists():
                    total_size = sum(f.stat().st_size for f in index_dir.rglob('*') if f.is_file())
                    return total_size / (1024 * 1024)
            elif indexer.datastore == 'DB1':
                if indexer.db_path.exists():
                    return indexer.db_path.stat().st_size / (1024 * 1024)
            elif indexer.datastore == 'DB2':
                if indexer.json_db_path.exists():
                    return indexer.json_db_path.stat().st_size / (1024 * 1024)
        except:
            pass
        return 0
    
    def _generate_all_plots(self):
        """Generate all required comparison plots"""
        
        results_dir = Path("selfindex_results")
        results_dir.mkdir(exist_ok=True)
        
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not successful_results:
            print("❌ No successful results to plot")
            return
        
        # Plot.C: Memory footprint for different index types (x=1,2,3)
        self._plot_metric_c_index_types(results_dir, successful_results)
        
        # Plot.A: Latency for different datastores (y=1,2,3)
        self._plot_metric_a_datastores(results_dir, successful_results)
        
        # Plot.AB: Throughput vs compression methods (z=1,2,3)
        self._plot_metric_ab_compression(results_dir, successful_results)
        
        # Plot.A: Latency with/without optimization (i=0,1)
        self._plot_metric_a_optimization(results_dir, successful_results)
        
        # Plot.AC: Memory vs query processing methods (q=T,D)
        self._plot_metric_ac_query_processing(results_dir, successful_results)
        
        # Overall comparison with ESIndex
        self._plot_overall_comparison(results_dir, successful_results)
        
        print(f"📊 All comparison plots saved to: {results_dir}")
    
    def _plot_metric_c_index_types(self, results_dir, successful_results):
        """Plot.C: Memory footprint for different index types"""
        
        index_types = ['BOOLEAN', 'WORDCOUNT', 'TFIDF']
        memory_usage = []
        index_sizes = []
        labels = []
        
        for idx_type in index_types:
            type_results = [r for r in successful_results.values() if r['config']['index_type'] == idx_type]
            if type_results:
                avg_memory = np.mean([r['memory_usage'] for r in type_results])
                avg_size = np.mean([r['index_size_mb'] for r in type_results])
                memory_usage.append(avg_memory)
                index_sizes.append(avg_size)
                labels.append(f"{idx_type}\n(x={index_types.index(idx_type)+1})")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Memory usage plot
        bars1 = ax1.bar(labels, memory_usage, color=['#ff9999', '#66b3ff', '#99ff99'])
        ax1.set_title('Plot.C: Memory Usage by Index Type', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_xlabel('Index Type')
        
        # Add ESIndex baseline
        es_memory = self.esindex_baseline['memory']['index_size_mb']
        ax1.axhline(y=es_memory, color='red', linestyle='--', label=f'ESIndex-v1.0: {es_memory}MB')
        ax1.legend()
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Index size plot
        bars2 = ax2.bar(labels, index_sizes, color=['#ff9999', '#66b3ff', '#99ff99'])
        ax2.set_title('Plot.C: Index Size by Index Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Index Size (MB)')
        ax2.set_xlabel('Index Type')
        
        # Add ESIndex baseline
        ax2.axhline(y=es_memory, color='red', linestyle='--', label=f'ESIndex-v1.0: {es_memory}MB')
        ax2.legend()
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(results_dir / "plot_c_index_types.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_a_datastores(self, results_dir, successful_results):
        """Plot.A: Latency comparison for different datastores"""
        
        datastores = ['CUSTOM', 'DB1', 'DB2']
        latencies = []
        labels = []
        
        for ds in datastores:
            ds_results = [r for r in successful_results.values() if r['config']['datastore'] == ds]
            if ds_results:
                avg_latency = np.mean([r['p95_query_time'] for r in ds_results if r['p95_query_time'] != float('inf')])
                latencies.append(avg_latency)
                labels.append(f"{ds}\n(y={datastores.index(ds)+1})")
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, latencies, color=['#ff7f0e', '#2ca02c', '#d62728'])
        plt.title('Plot.A: Query Latency by Datastore (P95)', fontsize=14, fontweight='bold')
        plt.ylabel('Latency (ms)')
        plt.xlabel('Datastore Type')
        
        # Add ESIndex baseline
        es_latency = self.esindex_baseline['latency']['p95']
        plt.axhline(y=es_latency, color='red', linestyle='--', label=f'ESIndex-v1.0: {es_latency}ms')
        plt.legend()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(results_dir / "plot_a_datastores.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_ab_compression(self, results_dir, successful_results):
        """Plot.AB: Throughput vs compression methods"""
        
        compressions = ['NONE', 'CODE', 'CLIB']
        throughputs = []
        index_sizes = []
        labels = []
        
        for comp in compressions:
            comp_results = [r for r in successful_results.values() if r['config']['compression'] == comp]
            if comp_results:
                avg_throughput = np.mean([r['throughput_qps'] for r in comp_results if r['throughput_qps'] > 0])
                avg_size = np.mean([r['index_size_mb'] for r in comp_results])
                throughputs.append(avg_throughput)
                index_sizes.append(avg_size)
                labels.append(f"{comp}\n(z={compressions.index(comp)+1})")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Throughput plot
        bars1 = ax1.bar(labels, throughputs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_title('Plot.AB: Throughput by Compression Method', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Throughput (queries/sec)')
        ax1.set_xlabel('Compression Method')
        
        # Add ESIndex baseline
        es_throughput = self.esindex_baseline['throughput']['single_thread_qps']
        ax1.axhline(y=es_throughput, color='red', linestyle='--', label=f'ESIndex-v1.0: {es_throughput:.1f} qps')
        ax1.legend()
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Index size plot
        bars2 = ax2.bar(labels, index_sizes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax2.set_title('Plot.AB: Index Size by Compression Method', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Index Size (MB)')
        ax2.set_xlabel('Compression Method')
        
        # Add ESIndex baseline
        es_size = self.esindex_baseline['memory']['index_size_mb']
        ax2.axhline(y=es_size, color='red', linestyle='--', label=f'ESIndex-v1.0: {es_size}MB')
        ax2.legend()
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(results_dir / "plot_ab_compression.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_a_optimization(self, results_dir, successful_results):
        """Plot.A: Latency with/without optimization"""
        
        optimizations = ['Null', 'Skipping']
        latencies = []
        labels = []
        
        for opt in optimizations:
            opt_results = [r for r in successful_results.values() if r['config']['optimization'] == opt]
            if opt_results:
                avg_latency = np.mean([r['p95_query_time'] for r in opt_results if r['p95_query_time'] != float('inf')])
                latencies.append(avg_latency)
                labels.append(f"{opt}\n(i={optimizations.index(opt)})")
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, latencies, color=['#9467bd', '#8c564b'])
        plt.title('Plot.A: Query Latency by Optimization (P95)', fontsize=14, fontweight='bold')
        plt.ylabel('Latency (ms)')
        plt.xlabel('Optimization Strategy')
        
        # Add ESIndex baseline
        es_latency = self.esindex_baseline['latency']['p95']
        plt.axhline(y=es_latency, color='red', linestyle='--', label=f'ESIndex-v1.0: {es_latency}ms')
        plt.legend()
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(results_dir / "plot_a_optimization.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_ac_query_processing(self, results_dir, successful_results):
        """Plot.AC: Memory vs query processing methods"""
        
        query_procs = ['TERMatat', 'DOCatat']
        memory_usage = []
        latencies = []
        labels = []
        
        for qp in query_procs:
            qp_results = [r for r in successful_results.values() if r['config']['query_proc'] == qp]
            if qp_results:
                avg_memory = np.mean([r['memory_usage'] for r in qp_results])
                avg_latency = np.mean([r['avg_query_time'] for r in qp_results if r['avg_query_time'] != float('inf')])
                memory_usage.append(avg_memory)
                latencies.append(avg_latency)
                labels.append(f"{'Term-at-a-time' if qp == 'TERMatat' else 'Doc-at-a-time'}\n(q={'T' if qp == 'TERMatat' else 'D'})")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Memory usage plot
        bars1 = ax1.bar(labels, memory_usage, color=['#e377c2', '#7f7f7f'])
        ax1.set_title('Plot.AC: Memory Usage by Query Processing', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_xlabel('Query Processing Method')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Latency plot
        bars2 = ax2.bar(labels, latencies, color=['#e377c2', '#7f7f7f'])
        ax2.set_title('Plot.AC: Query Latency by Processing Method', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_xlabel('Query Processing Method')
        
        # Add ESIndex baseline
        es_latency = self.esindex_baseline['latency']['mean']
        ax2.axhline(y=es_latency, color='red', linestyle='--', label=f'ESIndex-v1.0: {es_latency}ms')
        ax2.legend()
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(results_dir / "plot_ac_query_processing.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_overall_comparison(self, results_dir, successful_results):
        """Overall comparison between ESIndex and best SelfIndex variants"""
        
        if not successful_results:
            return
        
        # Find best performers
        best_latency = min(successful_results.items(), key=lambda x: x[1].get('avg_query_time', float('inf')))
        best_throughput = max(successful_results.items(), key=lambda x: x[1].get('throughput_qps', 0))
        best_memory = min(successful_results.items(), key=lambda x: x[1].get('index_size_mb', float('inf')))
        
        # Comparison data
        systems = ['ESIndex-v1.0', f'Best SelfIndex\n({best_latency[0][:15]}...)']
        latencies = [self.esindex_baseline['latency']['mean'], best_latency[1].get('avg_query_time', 0)]
        throughputs = [self.esindex_baseline['throughput']['single_thread_qps'], best_throughput[1].get('throughput_qps', 0)]
        memory_sizes = [self.esindex_baseline['memory']['index_size_mb'], best_memory[1].get('index_size_mb', 0)]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Latency comparison
        bars1 = ax1.bar(systems, latencies, color=['blue', 'orange'])
        ax1.set_title('Latency Comparison (Lower is Better)', fontweight='bold')
        ax1.set_ylabel('Latency (ms)')
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')
        
        # Throughput comparison
        bars2 = ax2.bar(systems, throughputs, color=['blue', 'green'])
        ax2.set_title('Throughput Comparison (Higher is Better)', fontweight='bold')
        ax2.set_ylabel('Queries per Second')
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')
        
        # Memory comparison
        bars3 = ax3.bar(systems, memory_sizes, color=['blue', 'red'])
        ax3.set_title('Index Size Comparison (Lower is Better)', fontweight='bold')
        ax3.set_ylabel('Index Size (MB)')
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')
        
        # Summary radar chart
        categories = ['Latency\n(norm)', 'Throughput', 'Memory\n(norm)', 'Overall']
        
        # Normalize values (higher is better)
        es_latency_norm = 100 / max(latencies[0], 1)
        self_latency_norm = 100 / max(latencies[1], 1)
        
        es_values = [es_latency_norm, throughputs[0], 100/max(memory_sizes[0], 1), 
                    (es_latency_norm + throughputs[0] + 100/max(memory_sizes[0], 1))/3]
        self_values = [self_latency_norm, throughputs[1], 100/max(memory_sizes[1], 1),
                      (self_latency_norm + throughputs[1] + 100/max(memory_sizes[1], 1))/3]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars4_1 = ax4.bar(x - width/2, es_values, width, label='ESIndex-v1.0', color='blue', alpha=0.7)
        bars4_2 = ax4.bar(x + width/2, self_values, width, label='Best SelfIndex', color='orange', alpha=0.7)
        
        ax4.set_title('Overall Performance Comparison\n(Normalized, Higher is Better)', fontweight='bold')
        ax4.set_ylabel('Normalized Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / "overall_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not successful_results:
            print("❌ No successful results for comparison report")
            return
        
        # Find best performers
        best_latency = min(successful_results.items(), key=lambda x: x[1].get('avg_query_time', float('inf')))
        best_throughput = max(successful_results.items(), key=lambda x: x[1].get('throughput_qps', 0))
        best_memory = min(successful_results.items(), key=lambda x: x[1].get('index_size_mb', float('inf')))
        
        report = {
            'evaluation_summary': {
                'total_configurations': len(self.results),
                'successful_tests': len(successful_results),
                'failed_tests': len(self.results) - len(successful_results)
            },
            'best_performers': {
                'latency': {
                    'config': best_latency[0],
                    'value': best_latency[1].get('avg_query_time', 0),
                    'config_details': best_latency[1]['config']
                },
                'throughput': {
                    'config': best_throughput[0],
                    'value': best_throughput[1].get('throughput_qps', 0),
                    'config_details': best_throughput[1]['config']
                },
                'memory': {
                    'config': best_memory[0],
                    'value': best_memory[1].get('index_size_mb', 0),
                    'config_details': best_memory[1]['config']
                }
            },
            'esindex_comparison': {
                'esindex_baseline': self.esindex_baseline,
                'best_selfindex_latency': best_latency[1].get('avg_query_time', 0),
                'best_selfindex_throughput': best_throughput[1].get('throughput_qps', 0),
                'best_selfindex_memory': best_memory[1].get('index_size_mb', 0)
            },
            'detailed_results': self.results
        }
        
        # Save report
        results_dir = Path("selfindex_results")
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "comprehensive_evaluation_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n📊 COMPARISON REPORT SUMMARY:")
        print(f"   🏆 Best Latency: {best_latency[0]} ({best_latency[1].get('avg_query_time', 0):.2f}ms)")
        print(f"   🚀 Best Throughput: {best_throughput[0]} ({best_throughput[1].get('throughput_qps', 0):.2f} qps)")
        print(f"   💾 Best Memory: {best_memory[0]} ({best_memory[1].get('index_size_mb', 0):.2f}MB)")
        
        print(f"\n📈 VS ESINDEX-V1.0:")
        print(f"   Latency: {best_latency[1].get('avg_query_time', 0):.2f}ms vs {self.esindex_baseline['latency']['mean']:.2f}ms")
        print(f"   Throughput: {best_throughput[1].get('throughput_qps', 0):.2f} vs {self.esindex_baseline['throughput']['single_thread_qps']:.2f} qps")
        print(f"   Memory: {best_memory[1].get('index_size_mb', 0):.2f}MB vs {self.esindex_baseline['memory']['index_size_mb']:.2f}MB")

def main():
    """Main function to run SelfIndex evaluation"""
    
    print("🚀 STARTING COMPREHENSIVE SELFINDEX EVALUATION")
    print("="*60)
    
    evaluator = SelfIndexEvaluator()
    results = evaluator.run_evaluation(max_configs=20, sample_size=100)
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Results saved to: selfindex_results/")
    print(f"   All required plots generated (Plot.A, Plot.AB, Plot.AC, Plot.C)")
    print(f"   Ready for comparison with ESIndex-v1.0")

if __name__ == "__main__":
    main()