#!/usr/bin/env python3
"""
Interactive demo of SelfIndex capabilities
"""

from self_index import SelfIndex
import json

def demo_single_index():
    """Demonstrate a single SelfIndex configuration"""
    
    print("🚀 SelfIndex Interactive Demo")
    print("="*40)
    
    # Create sample documents
    sample_docs = [
        ('doc1', 'Anarchism is a political philosophy and movement that is skeptical of all justifications for authority.'),
        ('doc2', 'Artificial intelligence is intelligence demonstrated by machines, in contrast to natural intelligence.'),
        ('doc3', 'Political philosophy examines the concepts and arguments involved in political thinking.'),
        ('doc4', 'Computer science is the study of algorithmic processes and computational systems.'),
        ('doc5', 'Social movements are organized efforts by groups of people to bring about or resist social change.')
    ]
    
    # Test different configurations
    configs = [
        {'name': 'Boolean + Custom', 'index_type': 'BOOLEAN', 'datastore': 'CUSTOM', 'compression': 'NONE'},
        {'name': 'TF-IDF + SQLite', 'index_type': 'TFIDF', 'datastore': 'DB1', 'compression': 'NONE'},
        {'name': 'WordCount + Compression', 'index_type': 'WORDCOUNT', 'datastore': 'CUSTOM', 'compression': 'CLIB'},
    ]
    
    test_queries = ['anarchism', 'artificial intelligence', 'political philosophy']
    
    for config in configs:
        print(f"\n📊 Testing: {config['name']}")
        print(f"   Configuration: {config}")
        
        try:
            # Create indexer
            indexer = SelfIndex(**{k: v for k, v in config.items() if k != 'name'})
            
            # Create index
            index_id = f"demo_{config['name'].replace(' ', '_').lower()}"
            indexer.create_index(index_id, sample_docs)
            
            # Test queries
            for query in test_queries:
                result = indexer.query(query)
                parsed = json.loads(result)
                
                if 'error' not in parsed:
                    print(f"   Query '{query}': {parsed['total_results']} results")
                else:
                    print(f"   Query '{query}': ERROR - {parsed['error']}")
            
            # Cleanup
            indexer.delete_index(index_id)
            print(f"   ✅ Success")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    demo_single_index()