#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import json
from self_index import SelfIndex
from datasets import load_dataset
from tqdm import tqdm
import time

def main():
    parser = argparse.ArgumentParser(description='SelfIndex CLI tool')
    parser.add_argument('action', choices=['create', 'query', 'list', 'delete', 'benchmark'],
                        help='Action to perform')
    parser.add_argument('--index-id', help='Index identifier')
    parser.add_argument('--index-type', choices=['BOOLEAN', 'WORDCOUNT', 'TFIDF'], default='BOOLEAN',
                       help='Type of index to create')
    parser.add_argument('--datastore', choices=['CUSTOM', 'DB1', 'DB2'], default='CUSTOM',
                       help='Datastore to use')
    parser.add_argument('--compression', choices=['NONE', 'CODE', 'CLIB'], default='NONE',
                       help='Compression method to use')
    parser.add_argument('--query-proc', choices=['TERMatat', 'DOCatat'], default='TERMatat',
                       help='Query processing method')
    parser.add_argument('--optimization', choices=['Null', 'Skipping'], default='Null',
                       help='Optimization strategy')
    parser.add_argument('--docs', type=int, default=1000,
                       help='Number of documents to process')
    parser.add_argument('--query', help='Query string for search')
    
    args = parser.parse_args()
    
    # Initialize SelfIndex with configuration
    indexer = SelfIndex(
        index_type=args.index_type,
        datastore=args.datastore,
        compression=args.compression,
        query_proc=args.query_proc,
        optimization=args.optimization
    )
    
    if args.action == 'list':
        indices = indexer.list_indices()
        print(f"Available indices ({len(indices)}):")
        for idx in indices:
            print(f"  - {idx}")
        return
        
    if args.action == 'delete':
        if not args.index_id:
            print("Error: --index-id is required for delete action")
            return
        indexer.delete_index(args.index_id)
        return
    
    if args.action == 'query':
        if not args.index_id:
            print("Error: --index-id is required for query action")
            return
        if not args.query:
            print("Error: --query is required for query action")
            return
            
        indexer.load_index(args.index_id)
        result = indexer.query(args.query)
        print(result)
        return
    
    if args.action == 'create':
        if not args.index_id:
            print("Error: --index-id is required for create action")
            return
            
        # Load data
        print(f"Loading data (limited to {args.docs} documents)...")
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
            print(f"Dataset loaded: {len(ds)} total documents available")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating sample data instead...")
            # Create sample data
            sample_data = []
            for i in range(min(args.docs, 100)):
                sample_data.append({
                    'id': f'sample_{i}',
                    'text': f'This is sample document {i}. It contains some test text for indexing purposes.',
                    'title': f'Sample Document {i}'
                })
            ds = sample_data
        
        # Prepare documents
        documents = []
        for idx, item in enumerate(tqdm(ds, desc="Preparing documents", total=min(args.docs, len(ds)))):
            if idx >= args.docs:
                break
                
            doc_id = str(item['id'])
            text = item['text']
            documents.append((doc_id, text))
        
        print(f"Creating index '{args.index_id}' with {len(documents)} documents...")
        indexer.create_index(args.index_id, documents)
        return
    
    if args.action == 'benchmark':
        if not args.index_id:
            print("Error: --index-id is required for benchmark action")
            return
            
        # Load data
        print(f"Loading data (limited to {args.docs} documents)...")
        try:
            ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
            print(f"Dataset loaded: {len(ds)} total documents available")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating sample data instead...")
            # Create sample data
            sample_data = []
            for i in range(min(args.docs, 100)):
                sample_data.append({
                    'id': f'sample_{i}',
                    'text': f'This is sample document {i}. It contains some test text for indexing purposes.',
                    'title': f'Sample Document {i}'
                })
            ds = sample_data
        
        # Prepare documents
        documents = []
        for idx, item in enumerate(tqdm(ds, desc="Preparing documents", total=min(args.docs, len(ds)))):
            if idx >= args.docs:
                break
                
            doc_id = str(item['id'])
            text = item['text']
            documents.append((doc_id, text))
        
        # Measure indexing time
        print(f"Benchmarking index creation for '{args.index_id}' with {len(documents)} documents...")
        start_time = time.time()
        indexer.create_index(args.index_id, documents)
        index_time = time.time() - start_time
        
        # Measure query time
        print("Benchmarking queries...")
        test_queries = [
            "philosophy", 
            "computer science", 
            "artificial intelligence", 
            "history of mathematics",
            "quantum physics theories"
        ]
        
        query_times = []
        for query in test_queries:
            start_time = time.time()
            result = indexer.query(query)
            query_time = (time.time() - start_time) * 1000  # ms
            query_times.append(query_time)
            print(f"  Query '{query}': {query_time:.2f} ms")
        
        # Print benchmark results
        print("\nBenchmark Results:")
        print(f"  Index Type: {args.index_type}")
        print(f"  Datastore: {args.datastore}")
        print(f"  Compression: {args.compression}")
        print(f"  Query Processing: {args.query_proc}")
        print(f"  Optimization: {args.optimization}")
        print(f"  Documents: {len(documents)}")
        print(f"  Indexing Time: {index_time:.2f} seconds")
        print(f"  Average Query Time: {sum(query_times)/len(query_times):.2f} ms")
        return

if __name__ == "__main__":
    main()