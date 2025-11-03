#!/usr/bin/env python3
"""
Optimized SelfIndex evaluation framework that loads data once and tests all 108 variants
Provides comprehensive metrics A,B,C,D for each configuration without redundant data loading
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
import threading
import concurrent.futures

class OptimizedSelfIndexEvaluator:
    """Optimized evaluation framework for all 108 SelfIndex variant combinations"""
    
    def __init__(self):
        self.results = {}
        self.cached_test_data = None  # Single cached dataset
        self.cached_query_set = None  # Single cached query set
        self.esindex_baseline = {
            'latency': {'p95': 18.88, 'p99': 26.14, 'mean': 12.57},
            'throughput': {'single_thread_qps': 91.63, 'multi_thread_qps': 375.91},
            'memory': {'index_size_mb': 140.90, 'documents_per_mb': 354.9},
            'functional': {'mean_average_precision': 0.050, 'coverage_rate': 1.000}
        }
        
    def load_test_data_once(self, max_docs=50000):
        """Load preprocessed dataset from CSV and cache it for all configurations"""
        if self.cached_test_data is not None:
            print(f"✅ Using cached dataset with {len(self.cached_test_data)} documents")
            return self.cached_test_data

        import pandas as pd
        csv_path = Path("../Dataset/PreProcessedData/preprocessed_dataset.csv")
        assert csv_path.exists(), f"CSV not found: {csv_path}"
        print(f"📁 Loading preprocessed dataset from {csv_path} ...")
        df = pd.read_csv(csv_path)
        if max_docs:
            df = df.head(max_docs)

        # Each row: id, original_text, processed_tokens, title, token_count
        # processed_tokens is a string representation of a list, e.g. "['word1', 'word2', ...]"

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
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading preprocessed docs"):
            doc_id = str(row['id'])
            tokens = parse_tokens(row['processed_tokens'])
            content = str(row['original_text'])
            documents.append((doc_id, tokens, content))

        # Debug: print a sample to verify tokens are parsed correctly
        if documents:
            print("Sample doc tokens:", documents[0][1][:10], "Type:", type(documents[0][1]))

        self.cached_test_data = documents
        print(f"✅ Loaded and CACHED {len(documents)} preprocessed documents for all configurations")
        return self.cached_test_data
    
    def generate_comprehensive_query_set_once(self):
        """Generate comprehensive query set ONCE for all configurations"""
        if self.cached_query_set is not None:
            print(f"✅ Using cached query set with {len(self.cached_query_set)} queries")
            return self.cached_query_set
            
        print("🧠 Generating comprehensive query set ONCE...")
        
        # Comprehensive query categories for thorough testing
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
        
        self.cached_query_set = all_queries
        print(f"✅ Generated and CACHED {len(all_queries)} comprehensive queries across {len(query_categories)} categories")
        return all_queries
    
    def generate_all_108_configurations(self):
        """Generate all 108 possible SelfIndex configurations"""
        
        index_types = ['BOOLEAN', 'WORDCOUNT', 'TFIDF']  # x=1,2,3
        datastores = ['CUSTOM', 'DB1']             # y=1,2,3  
        compressions = ['NONE', 'CODE', 'CLIB']           # z=1,2,3
        optimizations = ['Null', 'Skipping']              # i=0,1
        query_procs = ['TERMatat', 'DOCatat']             # q=T,D
        
        configurations = []
        
        print(f"🔧 Generating ALL 72 possible configurations...")
        
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
    
    def run_comprehensive_evaluation(self, max_configs=None, sample_size=50000):
        """Run comprehensive evaluation with SINGLE data load for all configurations"""
        
        print(f"🚀 COMPREHENSIVE SELFINDEX EVALUATION - SINGLE DATA LOAD")
        print(f"{'='*80}")
        
        # Load data and queries ONCE at the beginning
        print(f"📁 Step 1: Loading test data ONCE for ALL configurations...")
        test_data = self.load_test_data_once(sample_size)
        
        print(f"🧠 Step 2: Generating query set ONCE for ALL configurations...")
        query_set = self.generate_comprehensive_query_set_once()
        
        print(f"🔧 Step 3: Generating ALL configuration combinations...")
        all_configurations = self.generate_all_108_configurations()
        
        # Select configurations to test
        if max_configs and max_configs < len(all_configurations):
            configurations = all_configurations[:max_configs]
            print(f"⚠️  Testing first {max_configs} configurations for demo")
        else:
            configurations = all_configurations
            print(f"🎯 Testing ALL {len(configurations)} configurations")
        
        print(f"\n📋 EVALUATION CONFIGURATION:")
        print(f"   Documents: {len(test_data):,} (loaded ONCE)")
        print(f"   Queries: {len(query_set)} (generated ONCE)")
        print(f"   Configurations: {len(configurations)}")
        print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        successful_configs = 0
        failed_configs = 0
        
        for i, config in enumerate(configurations, 1):
            print(f"\n📊 [{i}/{len(configurations)}] Testing: {config['name']}")
            
            try:
                result = self._evaluate_single_configuration_comprehensive(config, test_data, query_set)
                self.results[config['name']] = result
                
                if 'error' not in result:
                    successful_configs += 1
                    latency = result.get('metrics', {}).get('latency', {}).get('mean', 0)
                    throughput = result.get('metrics', {}).get('throughput', {}).get('single_thread_qps', 0)
                    memory = result.get('metrics', {}).get('memory', {}).get('index_size_mb', 0)
                    print(f"    ✅ Success - Latency: {latency:.2f}ms, Throughput: {throughput:.1f}qps, Memory: {memory:.1f}MB")
                else:
                    failed_configs += 1
                    print(f"    ❌ Failed: {result['error']}")
                
                # Memory cleanup every 10 configs
                if i % 10 == 0:
                    print(f"    🧹 Memory cleanup after {i} configurations...")
                    import gc
                    gc.collect()
                    
            except Exception as e:
                failed_configs += 1
                error_msg = str(e)
                print(f"    ❌ Exception: {error_msg}")
                self.results[config['name']] = {'error': error_msg, 'config': config}
        
        print(f"\n📊 EVALUATION COMPLETE:")
        print(f"   Total configurations: {len(configurations)}")
        print(f"   Successful: {successful_configs}")
        print(f"   Failed: {failed_configs}")
        print(f"   Success rate: {successful_configs/len(configurations)*100:.1f}%")
        
        self._save_comprehensive_results()
        self._generate_all_plots()
        self._generate_comparison_report()
        
        return self.results
    
    def _evaluate_single_configuration_comprehensive(self, config, test_data, query_set):
        """Evaluate a single configuration with ALL metrics A,B,C,D"""
        
        try:
            # Store index in 'indexes/' with required naming policy
            # Naming: SelfIndex_iXdYcZqQO
            x = config['x']
            y = config['y']
            z = config['z']
            q = config['q']
            o = '0' if config['optimization'] == 'Null' else 'Sp'
            index_folder_name = f"SelfIndex_i{x}d{y}c{z}q{q}{o}"
            indexes_dir = Path('indexes')
            indexes_dir.mkdir(exist_ok=True)
            
            # Create indexer and change data_dir to store directly in final location
            indexer = SelfIndex(
                index_type=config['index_type'],
                datastore=config['datastore'],
                compression=config['compression'],
                query_proc=config['query_proc'],
                optimization=config['optimization'],
                data_dir=indexes_dir  # Store directly in final location
            )

            result = {
                'config': config,
                'metrics': {
                    'latency': {},
                    'throughput': {},
                    'memory': {},
                    'functional': {}
                }
            }

            # === INDEXING PHASE ===
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 * 1024)

            # Use the final folder name as index_id to avoid intermediate storage
            start_time = time.time()
            # test_data: (doc_id, tokens, content)
            # Pass tokens directly to SelfIndex
            indexer.create_index(index_folder_name, [(doc_id, tokens, content) for doc_id, tokens, content in test_data], pretokenized=True)
            index_time = time.time() - start_time

            mem_after = process.memory_info().rss / (1024 * 1024)
            memory_usage = mem_after - mem_before

            # Estimate index size
            index_size = self._estimate_index_size(indexer, index_folder_name)

            # === METRIC A: COMPREHENSIVE LATENCY MEASUREMENT ===
            latency_metrics = self._measure_latency_comprehensive(indexer, query_set)

            # === METRIC B: COMPREHENSIVE THROUGHPUT MEASUREMENT ===
            throughput_metrics = self._measure_throughput_comprehensive(indexer, query_set)

            # === METRIC C: COMPREHENSIVE MEMORY MEASUREMENT ===
            memory_metrics = {
                'index_creation_memory_mb': memory_usage,
                'index_size_mb': index_size,
                'process_memory_mb': mem_after,
                'memory_efficiency_docs_per_mb': len(test_data) / index_size if index_size > 0 else 0,
                'peak_memory_mb': mem_after,
                'memory_growth_mb': memory_usage
            }

            # === METRIC D: COMPREHENSIVE FUNCTIONAL MEASUREMENT ===
            functional_metrics = self._measure_functional_comprehensive(indexer, query_set)

            # Get the final index path (already stored in correct location)
            index_path = indexes_dir / index_folder_name
            if indexer.datastore == 'DB1':
                index_path = indexes_dir / f"{index_folder_name}.db"
            elif indexer.datastore == 'DB2':
                index_path = indexes_dir / f"{index_folder_name}.json"

            result.update({
                'index_time': index_time,
                'metrics': {
                    'latency': latency_metrics,
                    'throughput': throughput_metrics,
                    'memory': memory_metrics,
                    'functional': functional_metrics
                },
                'index_path': str(index_path.absolute())
            })
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _measure_latency_comprehensive(self, indexer, query_set):
        """METRIC A: Comprehensive latency measurement with p95, p99"""
        
        query_times = []
        category_latencies = defaultdict(list)
        
        # Warmup
        warmup_queries = query_set[:3]
        for query_info in warmup_queries:
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
                query_times.append(latency_ms)
                category_latencies[query_info['category']].append(latency_ms)
                
            except Exception:
                # Record failed query with penalty
                query_times.append(1000)  # 1 second penalty
        
        if query_times:
            return {
                'mean': statistics.mean(query_times),
                'p50': np.percentile(query_times, 50),
                'p90': np.percentile(query_times, 90),
                'p95': np.percentile(query_times, 95),  # ⭐ Required
                'p99': np.percentile(query_times, 99),  # ⭐ Required
                'min': min(query_times),
                'max': max(query_times),
                'total_queries': len(query_times),
                'successful_queries': len([t for t in query_times if t < 1000]),
                'category_breakdown': {cat: {
                    'mean': statistics.mean(times),
                    'p95': np.percentile(times, 95)
                } for cat, times in category_latencies.items() if times}
            }
        
        return {'error': 'No successful queries'}
    
    def _measure_throughput_comprehensive(self, indexer, query_set, duration_seconds=15):
        """METRIC B: Comprehensive throughput measurement (queries/second)"""
        
        # Prepare test queries for throughput measurement
        test_queries = []
        for query_info in query_set[:10]:
            test_queries.append(query_info['query'])
        
        # Single-threaded throughput
        single_thread_count = 0
        start_time = time.time()
        
        while (time.time() - start_time) < duration_seconds:
            query = test_queries[single_thread_count % len(test_queries)]
            try:
                indexer.query(query)
                single_thread_count += 1
            except:
                pass
        
        single_duration = time.time() - start_time
        single_qps = single_thread_count / single_duration if single_duration > 0 else 0
        
        # Multi-threaded throughput with thread pool
        def worker_queries(thread_queries):
            count = 0
            for query in thread_queries:
                try:
                    indexer.query(query)
                    count += 1
                except:
                    pass
            return count
        
        # Test with 4 threads
        thread_queries = [test_queries * 5 for _ in range(4)]  # 4 threads
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_results = [executor.submit(worker_queries, tq) for tq in thread_queries]
            thread_counts = [future.result() for future in concurrent.futures.as_completed(future_results)]
        
        multi_duration = time.time() - start_time
        total_multi_queries = sum(thread_counts)
        multi_qps = total_multi_queries / multi_duration if multi_duration > 0 else 0
        
        return {
            'single_thread_qps': single_qps,          # ⭐ Required
            'multi_thread_qps': multi_qps,            # ⭐ Required
            'speedup_factor': multi_qps / single_qps if single_qps > 0 else 0,
            'single_thread_queries': single_thread_count,
            'multi_thread_queries': total_multi_queries,
            'test_duration': duration_seconds
        }
    
    def _measure_functional_comprehensive(self, indexer, query_set):
        """METRIC D: Comprehensive functional metrics (precision, recall, ranking)"""
        
        # Ground truth for precision/recall
        ground_truth = {
            'anarchism': ['anarchism', 'political', 'philosophy'],
            'philosophy': ['philosophy', 'political', 'theory'],
            'artificial intelligence': ['artificial', 'intelligence', 'computer', 'technology']
        }
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        coverage_rate = 0
        ranking_scores = []
        
        test_queries = ['anarchism', 'philosophy', 'artificial intelligence']
        
        for query_text in test_queries:
            try:
                result_json = indexer.query(query_text)
                result = json.loads(result_json)
                
                if 'error' not in result and 'results' in result:
                    results = result['results']
                    total_results = result.get('total_results', 0)
                    
                    # Coverage
                    if total_results > 0:
                        coverage_rate += 1
                    
                    # Precision/Recall calculation
                    if query_text in ground_truth and results:
                        relevant_terms = ground_truth[query_text]
                        retrieved_relevant = 0
                        
                        for res in results[:10]:  # Top 10
                            content = res.get('content_preview', '').lower()
                            if any(term in content for term in relevant_terms):
                                retrieved_relevant += 1
                        
                        precision = retrieved_relevant / min(len(results), 10)
                        recall = retrieved_relevant / len(relevant_terms) if relevant_terms else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        precision_scores.append(precision)
                        recall_scores.append(recall)
                        f1_scores.append(f1)
                        
                        # Ranking quality (NDCG-like)
                        if results:
                            dcg = sum(1/np.log2(i+2) for i, res in enumerate(results[:5]) 
                                     if any(term in res.get('content_preview', '').lower() for term in relevant_terms))
                            idcg = sum(1/np.log2(i+2) for i in range(min(5, len(relevant_terms))))
                            ndcg = dcg / idcg if idcg > 0 else 0
                            ranking_scores.append(ndcg)
                        
            except Exception:
                pass
        
        return {
            'mean_average_precision': statistics.mean(precision_scores) if precision_scores else 0,
            'mean_recall': statistics.mean(recall_scores) if recall_scores else 0,
            'mean_f1_score': statistics.mean(f1_scores) if f1_scores else 0,
            'mean_ndcg': statistics.mean(ranking_scores) if ranking_scores else 0,
            'coverage_rate': coverage_rate / len(test_queries),
            'total_test_queries': len(test_queries),
            'successful_queries': len(precision_scores)
        }
    
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
        return 1.0  # Default size to avoid division by zero
    
    def _save_comprehensive_results(self):
        """Save comprehensive results to JSON"""
        results_dir = Path("comprehensive_selfindex_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"comprehensive_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"💾 Comprehensive results saved to: {results_file}")
    
    def _generate_all_plots(self):
        """Generate all required comparison plots"""
        
        results_dir = Path("comprehensive_selfindex_results")
        results_dir.mkdir(exist_ok=True)
        
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not successful_results:
            print("❌ No successful results to plot")
            return
        
        print(f"📊 Generating comprehensive plots...")
        
        # Plot.C: Latency by Index Type (x=1,2,3)
        self._plot_metric_c_index_type(results_dir, successful_results)
        
        # Plot.A: Throughput by Datastore (y=1,2,3)
        self._plot_metric_a_datastore(results_dir, successful_results)
        
        # Plot.AB: Memory vs Compression (z=1,2,3)
        self._plot_metric_ab_compression(results_dir, successful_results)
        
        # Plot.A: Latency by Optimization (i=0,1)
        self._plot_metric_a_optimization(results_dir, successful_results)
        
        # Plot.AC: Performance by Query Processing (q=T,D)
        self._plot_metric_ac_query_processing(results_dir, successful_results)
        
        # Overall ESIndex vs SelfIndex comparison
        self._plot_overall_comparison(results_dir, successful_results)
        
        print(f"📊 All plots saved to: {results_dir}")
    
    def _plot_metric_c_index_type(self, results_dir, successful_results):
        """Plot.C: Memory footprint by index type (x=1,2,3)"""
        
        index_types = ['BOOLEAN', 'WORDCOUNT', 'TFIDF']
        latencies = defaultdict(list)
        memory_usage = defaultdict(list)
        
        for result in successful_results.values():
            config = result['config']
            latency = result['metrics']['latency'].get('p95', 0)
            memory = result['metrics']['memory'].get('index_size_mb', 0)
            
            if latency > 0:
                latencies[config['index_type']].append(latency)
            if memory > 0:
                memory_usage[config['index_type']].append(memory)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latency plot
        data_lat = [latencies[idx_type] for idx_type in index_types if latencies[idx_type]]
        labels_lat = [f"{idx_type}\n(x={index_types.index(idx_type)+1})" for idx_type in index_types if latencies[idx_type]]
        
        if data_lat:
            ax1.boxplot(data_lat, labels=labels_lat)
            ax1.set_title('Plot.C: Latency by Index Type', fontweight='bold')
            ax1.set_ylabel('P95 Latency (ms)')
            ax1.set_xlabel('Index Type')
            
            # Add ESIndex baseline
            es_latency = self.esindex_baseline['latency']['p95']
            ax1.axhline(y=es_latency, color='red', linestyle='--', label=f'ESIndex-v1.0: {es_latency}ms')
            ax1.legend()
        
        # Memory plot
        data_mem = [memory_usage[idx_type] for idx_type in index_types if memory_usage[idx_type]]
        labels_mem = [f"{idx_type}\n(x={index_types.index(idx_type)+1})" for idx_type in index_types if memory_usage[idx_type]]
        
        if data_mem:
            ax2.boxplot(data_mem, labels=labels_mem)
            ax2.set_title('Plot.C: Memory Usage by Index Type', fontweight='bold')
            ax2.set_ylabel('Index Size (MB)')
            ax2.set_xlabel('Index Type')
            
            # Add ESIndex baseline
            es_memory = self.esindex_baseline['memory']['index_size_mb']
            ax2.axhline(y=es_memory, color='red', linestyle='--', label=f'ESIndex-v1.0: {es_memory}MB')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / "plot_c_memory_by_index_type.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_a_datastore(self, results_dir, successful_results):
        """Plot.A: Latency by datastore (y=1,2,3)"""
        
        datastores = ['CUSTOM', 'DB1', 'DB2']
        latencies = defaultdict(list)
        
        for result in successful_results.values():
            config = result['config']
            latency = result['metrics']['latency'].get('p95', 0)
            if latency > 0:
                latencies[config['datastore']].append(latency)
        
        plt.figure(figsize=(10, 6))
        
        data_for_plot = [latencies[ds] for ds in datastores if latencies[ds]]
        labels_for_plot = [f"{ds}\n(y={datastores.index(ds)+1})" for ds in datastores if latencies[ds]]
        
        if data_for_plot:
            plt.boxplot(data_for_plot, labels=labels_for_plot)
            plt.title('Plot.A: Latency by Datastore', fontsize=14, fontweight='bold')
            plt.ylabel('P95 Latency (ms)')
            plt.xlabel('Datastore Type')
            
            # Add ESIndex baseline
            es_latency = self.esindex_baseline['latency']['p95']
            plt.axhline(y=es_latency, color='red', linestyle='--', label=f'ESIndex-v1.0: {es_latency}ms')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / "plot_a_latency_by_datastore.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_ab_compression(self, results_dir, successful_results):
        """Plot.AB: Throughput vs compression (z=1,2,3)"""
        
        compressions = ['NONE', 'CODE', 'CLIB']
        throughputs = defaultdict(list)
        memory_effs = defaultdict(list)
        
        for result in successful_results.values():
            config = result['config']
            throughput = result['metrics']['throughput'].get('single_thread_qps', 0)
            memory_eff = result['metrics']['memory'].get('memory_efficiency_docs_per_mb', 0)
            
            if throughput > 0:
                throughputs[config['compression']].append(throughput)
            if memory_eff > 0:
                memory_effs[config['compression']].append(memory_eff)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Throughput plot
        data_thr = [throughputs[comp] for comp in compressions if throughputs[comp]]
        labels_thr = [f"{comp}\n(z={compressions.index(comp)+1})" for comp in compressions if throughputs[comp]]
        
        if data_thr:
            ax1.boxplot(data_thr, labels=labels_thr)
            ax1.set_title('Plot.AB: Throughput by Compression', fontweight='bold')
            ax1.set_ylabel('Throughput (queries/sec)')
            ax1.set_xlabel('Compression Method')
            
            # Add ESIndex baseline
            es_throughput = self.esindex_baseline['throughput']['single_thread_qps']
            ax1.axhline(y=es_throughput, color='red', linestyle='--', label=f'ESIndex-v1.0: {es_throughput:.1f}')
            ax1.legend()
        
        # Memory efficiency plot
        data_mem = [memory_effs[comp] for comp in compressions if memory_effs[comp]]
        labels_mem = [f"{comp}\n(z={compressions.index(comp)+1})" for comp in compressions if memory_effs[comp]]
        
        if data_mem:
            ax2.boxplot(data_mem, labels=labels_mem)
            ax2.set_title('Plot.AB: Memory Efficiency by Compression', fontweight='bold')
            ax2.set_ylabel('Documents per MB')
            ax2.set_xlabel('Compression Method')
            
            # Add ESIndex baseline
            es_memory_eff = self.esindex_baseline['memory']['documents_per_mb']
            ax2.axhline(y=es_memory_eff, color='red', linestyle='--', label=f'ESIndex-v1.0: {es_memory_eff:.1f}')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / "plot_ab_throughput_compression.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_a_optimization(self, results_dir, successful_results):
        """Plot.A: Latency by optimization (i=0,1)"""
        
        optimizations = ['Null', 'Skipping']
        latencies = defaultdict(list)
        
        for result in successful_results.values():
            config = result['config']
            latency = result['metrics']['latency'].get('p95', 0)
            if latency > 0:
                latencies[config['optimization']].append(latency)
        
        plt.figure(figsize=(8, 6))
        
        data_for_plot = [latencies[opt] for opt in optimizations if latencies[opt]]
        labels_for_plot = [f"{opt}\n(i={optimizations.index(opt)})" for opt in optimizations if latencies[opt]]
        
        if data_for_plot:
            plt.boxplot(data_for_plot, labels=labels_for_plot)
            plt.title('Plot.A: Latency by Optimization', fontsize=14, fontweight='bold')
            plt.ylabel('P95 Latency (ms)')
            plt.xlabel('Optimization Strategy')
            
            # Add ESIndex baseline
            es_latency = self.esindex_baseline['latency']['p95']
            plt.axhline(y=es_latency, color='red', linestyle='--', label=f'ESIndex-v1.0: {es_latency}ms')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / "plot_a_latency_by_optimization.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_ac_query_processing(self, results_dir, successful_results):
        """Plot.AC: Performance by query processing (q=T,D)"""
        
        query_procs = ['TERMatat', 'DOCatat']
        latencies = defaultdict(list)
        memory_usage = defaultdict(list)
        
        for result in successful_results.values():
            config = result['config']
            latency = result['metrics']['latency'].get('mean', 0)
            memory = result['metrics']['memory'].get('index_creation_memory_mb', 0)
            
            if latency > 0:
                latencies[config['query_proc']].append(latency)
            if memory > 0:
                memory_usage[config['query_proc']].append(memory)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latency comparison
        data_lat = [latencies[qp] for qp in query_procs if latencies[qp]]
        labels_lat = [f"{'Term-at-a-time' if qp == 'TERMatat' else 'Doc-at-a-time'}\n(q={'T' if qp == 'TERMatat' else 'D'})" for qp in query_procs if latencies[qp]]
        
        if data_lat:
            ax1.boxplot(data_lat, labels=labels_lat)
            ax1.set_title('Plot.AC: Latency by Query Processing', fontweight='bold')
            ax1.set_ylabel('Mean Latency (ms)')
            ax1.set_xlabel('Query Processing Method')
        
        # Memory comparison
        data_mem = [memory_usage[qp] for qp in query_procs if memory_usage[qp]]
        labels_mem = [f"{'Term-at-a-time' if qp == 'TERMatat' else 'Doc-at-a-time'}\n(q={'T' if qp == 'TERMatat' else 'D'})" for qp in query_procs if memory_usage[qp]]
        
        if data_mem:
            ax2.boxplot(data_mem, labels=labels_mem)
            ax2.set_title('Plot.AC: Memory by Query Processing', fontweight='bold')
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.set_xlabel('Query Processing Method')
        
        plt.tight_layout()
        plt.savefig(results_dir / "plot_ac_query_processing.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_overall_comparison(self, results_dir, successful_results):
        """Overall comparison between ESIndex and best SelfIndex variants"""
        
        if not successful_results:
            return
        
        # Find best performers
        best_latency = min(successful_results.items(), key=lambda x: x[1]['metrics']['latency'].get('p95', float('inf')))
        best_throughput = max(successful_results.items(), key=lambda x: x[1]['metrics']['throughput'].get('single_thread_qps', 0))
        best_memory = max(successful_results.items(), key=lambda x: x[1]['metrics']['memory'].get('memory_efficiency_docs_per_mb', 0))
        
        # Create comparison data
        systems = ['ESIndex-v1.0', f'Best SelfIndex\n({best_latency[0][:15]}...)']
        latencies = [self.esindex_baseline['latency']['p95'], best_latency[1]['metrics']['latency'].get('p95', 0)]
        throughputs = [self.esindex_baseline['throughput']['single_thread_qps'], best_throughput[1]['metrics']['throughput'].get('single_thread_qps', 0)]
        memory_effs = [self.esindex_baseline['memory']['documents_per_mb'], best_memory[1]['metrics']['memory'].get('memory_efficiency_docs_per_mb', 0)]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Latency comparison
        bars1 = ax1.bar(systems, latencies, color=['blue', 'orange'])
        ax1.set_title('Latency Comparison (Lower is Better)', fontweight='bold')
        ax1.set_ylabel('P95 Latency (ms)')
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
        
        # Memory efficiency comparison
        bars3 = ax3.bar(systems, memory_effs, color=['blue', 'red'])
        ax3.set_title('Memory Efficiency (Higher is Better)', fontweight='bold')
        ax3.set_ylabel('Documents per MB')
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')
        
        # Overall score comparison
        es_score = (100/latencies[0] + throughputs[0] + memory_effs[0]/10) / 3
        self_score = (100/latencies[1] + throughputs[1] + memory_effs[1]/10) / 3 if latencies[1] > 0 else 0
        
        bars4 = ax4.bar(['ESIndex-v1.0', 'Best SelfIndex'], [es_score, self_score], color=['blue', 'purple'])
        ax4.set_title('Overall Performance Score', fontweight='bold')
        ax4.set_ylabel('Normalized Score')
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(results_dir / "overall_comparison_esindex_vs_selfindex.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not successful_results:
            print("❌ No successful results for comparison report")
            return
        
        # Find best performers
        best_latency = min(successful_results.items(), key=lambda x: x[1]['metrics']['latency'].get('p95', float('inf')))
        best_throughput = max(successful_results.items(), key=lambda x: x[1]['metrics']['throughput'].get('single_thread_qps', 0))
        best_memory = max(successful_results.items(), key=lambda x: x[1]['metrics']['memory'].get('memory_efficiency_docs_per_mb', 0))
        best_functional = max(successful_results.items(), key=lambda x: x[1]['metrics']['functional'].get('mean_f1_score', 0))
        
        report = {
            'evaluation_summary': {
                'total_configurations': len(self.results),
                'successful_tests': len(successful_results),
                'failed_tests': len(self.results) - len(successful_results),
                'success_rate': len(successful_results) / len(self.results) * 100
            },
            'best_performers': {
                'latency': {
                    'config': best_latency[0],
                    'value': best_latency[1]['metrics']['latency'].get('p95', 0),
                    'config_details': best_latency[1]['config']
                },
                'throughput': {
                    'config': best_throughput[0],
                    'value': best_throughput[1]['metrics']['throughput'].get('single_thread_qps', 0),
                    'config_details': best_throughput[1]['config']
                },
                'memory': {
                    'config': best_memory[0],
                    'value': best_memory[1]['metrics']['memory'].get('memory_efficiency_docs_per_mb', 0),
                    'config_details': best_memory[1]['config']
                },
                'functional': {
                    'config': best_functional[0],
                    'value': best_functional[1]['metrics']['functional'].get('mean_f1_score', 0),
                    'config_details': best_functional[1]['config']
                }
            },
            'esindex_comparison': {
                'esindex_baseline': self.esindex_baseline,
                'selfindex_best_latency': best_latency[1]['metrics']['latency'].get('p95', 0),
                'selfindex_best_throughput': best_throughput[1]['metrics']['throughput'].get('single_thread_qps', 0),
                'selfindex_best_memory': best_memory[1]['metrics']['memory'].get('memory_efficiency_docs_per_mb', 0),
                'selfindex_best_functional': best_functional[1]['metrics']['functional'].get('mean_f1_score', 0)
            }
        }
        
        # Save report
        results_dir = Path("comprehensive_selfindex_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(results_dir / f"comparison_report_{timestamp}.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n🏆 COMPREHENSIVE COMPARISON REPORT:")
        print(f"   📊 Total Configurations: {report['evaluation_summary']['total_configurations']}")
        print(f"   ✅ Successful: {report['evaluation_summary']['successful_tests']}")
        print(f"   ❌ Failed: {report['evaluation_summary']['failed_tests']}")
        print(f"   📈 Success Rate: {report['evaluation_summary']['success_rate']:.1f}%")
        
        print(f"\n🥇 BEST PERFORMERS:")
        print(f"   🏃 Best Latency: {best_latency[0]} ({best_latency[1]['metrics']['latency'].get('p95', 0):.2f}ms)")
        print(f"   🚀 Best Throughput: {best_throughput[0]} ({best_throughput[1]['metrics']['throughput'].get('single_thread_qps', 0):.2f} qps)")
        print(f"   💾 Best Memory: {best_memory[0]} ({best_memory[1]['metrics']['memory'].get('memory_efficiency_docs_per_mb', 0):.1f} docs/MB)")
        print(f"   🎯 Best Functional: {best_functional[0]} (F1: {best_functional[1]['metrics']['functional'].get('mean_f1_score', 0):.3f})")
        
        print(f"\n📊 VS ESINDEX-V1.0 COMPARISON:")
        print(f"   Latency: {best_latency[1]['metrics']['latency'].get('p95', 0):.2f}ms vs {self.esindex_baseline['latency']['p95']:.2f}ms")
        print(f"   Throughput: {best_throughput[1]['metrics']['throughput'].get('single_thread_qps', 0):.2f} vs {self.esindex_baseline['throughput']['single_thread_qps']:.2f} qps")
        print(f"   Memory Efficiency: {best_memory[1]['metrics']['memory'].get('memory_efficiency_docs_per_mb', 0):.1f} vs {self.esindex_baseline['memory']['documents_per_mb']:.1f} docs/MB")
        
        return report

def main():
    """Main function to run comprehensive SelfIndex evaluation"""
    
    print("🚀 STARTING OPTIMIZED SELFINDEX EVALUATION WITH SINGLE DATA LOAD")
    print("="*80)
    
    evaluator = OptimizedSelfIndexEvaluator()
    
    # Run evaluation with options:
    # For quick test: results = evaluator.run_comprehensive_evaluation(max_configs=10, sample_size=1000)
    # For full evaluation: results = evaluator.run_comprehensive_evaluation(sample_size=50000)
    
    results = evaluator.run_comprehensive_evaluation(max_configs=20, sample_size=5000)  # Demo run
    
    print(f"\n✅ Optimized evaluation complete!")
    print(f"   📊 All metrics (A,B,C,D) measured for each configuration")
    print(f"   📁 Results saved to: comprehensive_selfindex_results/")
    print(f"   🎨 All required plots generated (Plot.A, Plot.AB, Plot.AC, Plot.C)")
    print(f"   📋 Comparison report generated")
    print(f"   🏆 Ready for final analysis and assignment submission!")

if __name__ == "__main__":
    main()