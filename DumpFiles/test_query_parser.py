"""
Unit tests for Boolean query parser.
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from query_parser import QueryParser, QueryToken, TermNode, PhraseNode, AndNode, OrNode, NotNode
from query_parser import BooleanQueryExecutor
from self_index import SelfIndex, create_self_index


class TestQueryTokenizer(unittest.TestCase):
    """Test query tokenization."""
    
    def setUp(self):
        self.parser = QueryParser()
    
    def test_tokenize_simple_term(self):
        """Test tokenizing a simple quoted term."""
        tokens = self.parser.tokenize('"apple"')
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, 'TERM')
        self.assertEqual(tokens[0].value, 'apple')
    
    def test_tokenize_phrase(self):
        """Test tokenizing a phrase."""
        tokens = self.parser.tokenize('"quick brown fox"')
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].type, 'PHRASE')
        self.assertEqual(tokens[0].value, 'quick brown fox')
    
    def test_tokenize_and_operator(self):
        """Test tokenizing AND operator."""
        tokens = self.parser.tokenize('"apple" AND "banana"')
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[0].type, 'TERM')
        self.assertEqual(tokens[1].type, 'AND')
        self.assertEqual(tokens[2].type, 'TERM')
    
    def test_tokenize_or_operator(self):
        """Test tokenizing OR operator."""
        tokens = self.parser.tokenize('"apple" OR "banana"')
        self.assertEqual(len(tokens), 3)
        self.assertEqual(tokens[1].type, 'OR')
    
    def test_tokenize_not_operator(self):
        """Test tokenizing NOT operator."""
        tokens = self.parser.tokenize('NOT "apple"')
        self.assertEqual(len(tokens), 2)
        self.assertEqual(tokens[0].type, 'NOT')
        self.assertEqual(tokens[1].type, 'TERM')
    
    def test_tokenize_parentheses(self):
        """Test tokenizing parentheses."""
        tokens = self.parser.tokenize('("apple" AND "banana")')
        self.assertEqual(tokens[0].type, 'LPAREN')
        self.assertEqual(tokens[-1].type, 'RPAREN')
    
    def test_tokenize_complex_query(self):
        """Test tokenizing a complex query."""
        tokens = self.parser.tokenize('("apple" OR "banana") AND NOT "cherry"')
        self.assertGreater(len(tokens), 5)


class TestQueryParser(unittest.TestCase):
    """Test query parsing."""
    
    def setUp(self):
        self.parser = QueryParser()
    
    def test_parse_single_term(self):
        """Test parsing a single term."""
        ast = self.parser.parse('"apple"')
        self.assertIsInstance(ast, TermNode)
        self.assertEqual(ast.term, 'apple')
    
    def test_parse_phrase(self):
        """Test parsing a phrase."""
        ast = self.parser.parse('"quick brown fox"')
        self.assertIsInstance(ast, PhraseNode)
        self.assertEqual(ast.phrase, 'quick brown fox')
    
    def test_parse_and(self):
        """Test parsing AND expression."""
        ast = self.parser.parse('"apple" AND "banana"')
        self.assertIsInstance(ast, AndNode)
        self.assertIsInstance(ast.left, TermNode)
        self.assertIsInstance(ast.right, TermNode)
    
    def test_parse_or(self):
        """Test parsing OR expression."""
        ast = self.parser.parse('"apple" OR "banana"')
        self.assertIsInstance(ast, OrNode)
    
    def test_parse_not(self):
        """Test parsing NOT expression."""
        ast = self.parser.parse('NOT "apple"')
        self.assertIsInstance(ast, NotNode)
        self.assertIsInstance(ast.operand, TermNode)
    
    def test_parse_parentheses(self):
        """Test parsing with parentheses."""
        ast = self.parser.parse('("apple" OR "banana") AND "cherry"')
        self.assertIsInstance(ast, AndNode)
        self.assertIsInstance(ast.left, OrNode)
    
    def test_operator_precedence(self):
        """Test correct operator precedence: NOT > AND > OR."""
        # "apple" OR "banana" AND NOT "cherry" should be parsed as:
        # "apple" OR ("banana" AND (NOT "cherry"))
        ast = self.parser.parse('"apple" OR "banana" AND NOT "cherry"')
        self.assertIsInstance(ast, OrNode)
        self.assertIsInstance(ast.right, AndNode)
        self.assertIsInstance(ast.right.right, NotNode)


class TestQueryExecution(unittest.TestCase):
    """Test query execution against an index."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.original_storage_dir = SelfIndex.INDEX_STORAGE_DIR
        SelfIndex.INDEX_STORAGE_DIR = Path(self.test_dir)
        
        # Create test documents
        self.test_docs = [
            ("doc1", "apple banana cherry"),
            ("doc2", "banana cherry date"),
            ("doc3", "cherry date elderberry"),
            ("doc4", "apple date"),
        ]
        
        self.index = create_self_index('test_query', self.test_docs, info='BOOLEAN')
    
    def tearDown(self):
        """Clean up test fixtures."""
        SelfIndex.INDEX_STORAGE_DIR = self.original_storage_dir
        shutil.rmtree(self.test_dir, ignore_errors=True)
        SelfIndex._loaded_indices.clear()
    
    def test_execute_single_term(self):
        """Test executing a single term query."""
        executor = BooleanQueryExecutor(self.index)
        matching_docs, results = executor.execute('"apple"')
        
        self.assertIn('doc1', matching_docs)
        self.assertIn('doc4', matching_docs)
        self.assertEqual(len(matching_docs), 2)
    
    def test_execute_and_query(self):
        """Test executing an AND query."""
        executor = BooleanQueryExecutor(self.index)
        matching_docs, results = executor.execute('"apple" AND "banana"')
        
        # Only doc1 has both apple and banana
        self.assertEqual(len(matching_docs), 1)
        self.assertIn('doc1', matching_docs)
    
    def test_execute_or_query(self):
        """Test executing an OR query."""
        executor = BooleanQueryExecutor(self.index)
        matching_docs, results = executor.execute('"apple" OR "elderberry"')
        
        # doc1, doc4 have apple; doc3 has elderberry
        self.assertEqual(len(matching_docs), 3)
        self.assertIn('doc1', matching_docs)
        self.assertIn('doc4', matching_docs)
        self.assertIn('doc3', matching_docs)
    
    def test_execute_not_query(self):
        """Test executing a NOT query."""
        executor = BooleanQueryExecutor(self.index)
        matching_docs, results = executor.execute('NOT "apple"')
        
        # doc2 and doc3 don't have apple
        self.assertIn('doc2', matching_docs)
        self.assertIn('doc3', matching_docs)
        self.assertNotIn('doc1', matching_docs)
        self.assertNotIn('doc4', matching_docs)
    
    def test_execute_complex_query(self):
        """Test executing a complex Boolean query."""
        executor = BooleanQueryExecutor(self.index)
        # Find documents with (apple OR banana) AND cherry
        matching_docs, results = executor.execute('("apple" OR "banana") AND "cherry"')
        
        # doc1 has apple and cherry
        # doc2 has banana and cherry
        self.assertEqual(len(matching_docs), 2)
        self.assertIn('doc1', matching_docs)
        self.assertIn('doc2', matching_docs)
    
    def test_phrase_query(self):
        """Test phrase query."""
        # Create documents with specific phrases
        docs = [
            ("doc1", "the quick brown fox jumps"),
            ("doc2", "quick brown is a color"),
            ("doc3", "the fox is quick and brown"),
        ]
        index = create_self_index('test_phrase', docs, info='BOOLEAN')
        
        executor = BooleanQueryExecutor(index)
        matching_docs, results = executor.execute('"quick brown"')
        
        # Only doc1 and doc2 have "quick brown" as a phrase
        self.assertIn('doc1', matching_docs)
        self.assertIn('doc2', matching_docs)
        self.assertNotIn('doc3', matching_docs)  # has both words but not adjacent


class TestQueryIntegration(unittest.TestCase):
    """Test integration of query parser with SelfIndex."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.original_storage_dir = SelfIndex.INDEX_STORAGE_DIR
        SelfIndex.INDEX_STORAGE_DIR = Path(self.test_dir)
        
        self.test_docs = [
            ("doc1", "apple banana cherry"),
            ("doc2", "banana cherry date"),
            ("doc3", "cherry date elderberry"),
        ]
        
        self.index = create_self_index('test_integration', self.test_docs, info='BOOLEAN')
    
    def tearDown(self):
        """Clean up test fixtures."""
        SelfIndex.INDEX_STORAGE_DIR = self.original_storage_dir
        shutil.rmtree(self.test_dir, ignore_errors=True)
        SelfIndex._loaded_indices.clear()
    
    def test_query_through_index(self):
        """Test querying through SelfIndex.query() method."""
        import json
        
        # Execute Boolean query through index
        result_json = self.index.query('"apple" AND "banana"')
        result = json.loads(result_json)
        
        # Check result structure
        self.assertIn('query', result)
        self.assertIn('num_results', result)
        self.assertIn('results', result)
        
        # Only doc1 should match
        self.assertEqual(result['num_results'], 1)
        if result['num_results'] > 0:
            self.assertEqual(result['results'][0]['doc_id'], 'doc1')


if __name__ == '__main__':
    unittest.main()
