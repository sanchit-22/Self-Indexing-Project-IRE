#!/usr/bin/env python3
"""
Final verification script for SelfIndex-v1.0 implementation.

This script verifies that all components work correctly.
"""

import sys
import json
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from self_index import SelfIndex, create_self_index
        from query_parser import QueryParser, BooleanQueryExecutor
        from compression import get_compressor
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic SelfIndex functionality."""
    print("\nTesting basic functionality...")
    try:
        from self_index import create_self_index
        
        # Create a simple index
        docs = [
            ("doc1", "quick brown fox"),
            ("doc2", "lazy brown dog"),
        ]
        
        index = create_self_index('verify_test', docs, info='BOOLEAN')
        
        # Query it
        result = json.loads(index.query('"brown"'))
        
        if result['num_results'] == 2:
            print("  ✓ Basic query works")
            
            # Clean up
            index.delete_index('verify_test')
            return True
        else:
            print(f"  ✗ Expected 2 results, got {result['num_results']}")
            return False
            
    except Exception as e:
        print(f"  ✗ Basic functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_boolean_queries():
    """Test Boolean query functionality."""
    print("\nTesting Boolean queries...")
    try:
        from self_index import create_self_index
        
        docs = [
            ("doc1", "apple banana"),
            ("doc2", "banana cherry"),
            ("doc3", "cherry date"),
        ]
        
        index = create_self_index('verify_bool', docs, info='BOOLEAN')
        
        tests = [
            ('"apple" AND "banana"', 1),
            ('"banana" OR "date"', 3),
            ('NOT "apple"', 2),
        ]
        
        all_pass = True
        for query, expected in tests:
            result = json.loads(index.query(query))
            if result['num_results'] == expected:
                print(f"  ✓ {query} -> {expected} results")
            else:
                print(f"  ✗ {query} -> expected {expected}, got {result['num_results']}")
                all_pass = False
        
        index.delete_index('verify_bool')
        return all_pass
        
    except Exception as e:
        print(f"  ✗ Boolean queries failed: {e}")
        return False

def test_compression():
    """Test compression functionality."""
    print("\nTesting compression...")
    try:
        from compression import get_compressor
        
        positions = [0, 1, 2, 5, 10, 50, 100]
        
        for comp_type in ['NONE', 'CODE', 'CLIB']:
            compressor = get_compressor(comp_type)
            compressed = compressor.compress_posting_list(positions)
            decompressed = compressor.decompress_posting_list(compressed)
            
            if decompressed == positions:
                print(f"  ✓ {comp_type} compression works")
            else:
                print(f"  ✗ {comp_type} compression failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Compression failed: {e}")
        return False

def test_all_index_types():
    """Test all index types."""
    print("\nTesting all index types...")
    try:
        from self_index import create_self_index
        
        docs = [
            ("doc1", "search query"),
            ("doc2", "search results"),
        ]
        
        for info_type in ['BOOLEAN', 'WORDCOUNT', 'TFIDF']:
            index = create_self_index(f'verify_{info_type.lower()}', 
                                     docs, info=info_type)
            result = json.loads(index.query('"search"'))
            
            if result['num_results'] == 2:
                print(f"  ✓ {info_type} index works")
                index.delete_index(f'verify_{info_type.lower()}')
            else:
                print(f"  ✗ {info_type} index failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Index types failed: {e}")
        return False

def run_unit_tests():
    """Run the unit tests."""
    print("\nRunning unit tests...")
    try:
        import unittest
        import sys
        from io import StringIO
        
        # Capture test output
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover('.', pattern='test_*.py')
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=0, stream=StringIO())
        result = runner.run(test_suite)
        
        if result.wasSuccessful():
            print(f"  ✓ All {result.testsRun} unit tests passed")
            return True
        else:
            print(f"  ✗ {len(result.failures)} failures, {len(result.errors)} errors")
            return False
            
    except Exception as e:
        print(f"  ✗ Unit tests failed: {e}")
        return False

def check_files():
    """Check that all expected files exist."""
    print("\nChecking files...")
    expected_files = [
        'self_index.py',
        'query_parser.py',
        'compression.py',
        'test_self_index.py',
        'test_query_parser.py',
        'test_compression.py',
        'example_usage.py',
        'performance_eval.py',
        'README_SelfIndex.md',
        'IMPLEMENTATION_SUMMARY.md',
    ]
    
    all_exist = True
    for filename in expected_files:
        if Path(filename).exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all verification tests."""
    print("="*70)
    print("SelfIndex-v1.0 Verification Script")
    print("="*70)
    
    tests = [
        ("File Check", check_files),
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Boolean Queries", test_boolean_queries),
        ("Compression", test_compression),
        ("All Index Types", test_all_index_types),
        ("Unit Tests", run_unit_tests),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ✗ {name} crashed: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "="*70)
    print(f"Results: {passed_count}/{total_count} tests passed")
    print("="*70)
    
    if passed_count == total_count:
        print("\n✓ ALL VERIFICATIONS PASSED - Implementation is complete!")
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} verification(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
