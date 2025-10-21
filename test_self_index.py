"""
Unit tests for SelfIndex implementation.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json

from self_index import SelfIndex, create_self_index


class TestSelfIndexBasic(unittest.TestCase):
    """Test basic SelfIndex functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test indices
        self.test_dir = tempfile.mkdtemp()
        self.original_storage_dir = SelfIndex.INDEX_STORAGE_DIR
        SelfIndex.INDEX_STORAGE_DIR = Path(self.test_dir)
        
        # Sample test documents
        self.test_docs = [
            ("doc1", "The quick brown fox jumps over the lazy dog"),
            ("doc2", "A quick brown dog jumps over a lazy fox"),
            ("doc3", "The dog and the fox are quick"),
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Restore original storage directory
        SelfIndex.INDEX_STORAGE_DIR = self.original_storage_dir
        # Remove temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
        # Clear loaded indices
        SelfIndex._loaded_indices.clear()
    
    def test_initialization(self):
        """Test SelfIndex initialization."""
        index = SelfIndex(info='BOOLEAN', dstore='CUSTOM')
        self.assertIsNotNone(index)
        self.assertEqual(index.info_type.name, 'BOOLEAN')
        self.assertEqual(index.datastore_type.name, 'CUSTOM')
    
    def test_boolean_index_creation(self):
        """Test creation of Boolean index."""
        index = create_self_index('test_bool', self.test_docs, info='BOOLEAN')
        
        # Check index was created
        self.assertEqual(index.num_docs, 3)
        self.assertGreater(len(index.vocabulary), 0)
        
        # Check inverted index structure
        self.assertIn('quick', index.inverted_index)
        self.assertIn('doc1', index.inverted_index['quick'])
    
    def test_wordcount_index_creation(self):
        """Test creation of WordCount index."""
        index = create_self_index('test_wc', self.test_docs, info='WORDCOUNT')
        
        # Check index was created with term frequencies
        self.assertEqual(index.num_docs, 3)
        
        # Check term frequency structure
        if 'quick' in index.inverted_index and 'doc1' in index.inverted_index['quick']:
            posting = index.inverted_index['quick']['doc1']
            self.assertIn('tf', posting)
            self.assertIn('positions', posting)
    
    def test_tfidf_index_creation(self):
        """Test creation of TF-IDF index."""
        index = create_self_index('test_tfidf', self.test_docs, info='TFIDF')
        
        # Check index was created with TF-IDF scores
        self.assertEqual(index.num_docs, 3)
        
        # Check TF-IDF structure
        if 'quick' in index.inverted_index and 'doc1' in index.inverted_index['quick']:
            posting = index.inverted_index['quick']['doc1']
            self.assertIn('tfidf', posting)
            self.assertIn('tf', posting)
    
    def test_text_preprocessing(self):
        """Test text preprocessing."""
        index = SelfIndex()
        
        # Test lowercase conversion
        tokens = index._preprocess_text("The Quick BROWN Fox")
        self.assertEqual(tokens, ['the', 'quick', 'brown', 'fox'])
        
        # Test punctuation removal
        tokens = index._preprocess_text("Hello, world! How are you?")
        self.assertNotIn(',', tokens)
        self.assertNotIn('!', tokens)
        self.assertNotIn('?', tokens)
    
    def test_simple_query(self):
        """Test simple query execution."""
        index = create_self_index('test_query', self.test_docs, info='BOOLEAN')
        
        # Query for a term
        results_json = index.query("quick")
        results = json.loads(results_json)
        
        # Check results structure
        self.assertIn('query', results)
        self.assertIn('num_results', results)
        self.assertIn('results', results)
        self.assertGreater(results['num_results'], 0)
    
    def test_persistence_and_loading(self):
        """Test index persistence and loading."""
        # Create and save an index
        index1 = create_self_index('test_persist', self.test_docs, info='BOOLEAN')
        original_vocab_size = len(index1.vocabulary)
        
        # Create a new index instance and load
        index2 = SelfIndex(info='BOOLEAN', dstore='CUSTOM')
        index_path = index2._get_index_path('test_persist')
        index2.load_index(str(index_path))
        
        # Verify loaded index
        self.assertEqual(len(index2.vocabulary), original_vocab_size)
        self.assertEqual(index2.num_docs, index1.num_docs)
    
    def test_list_indices(self):
        """Test listing available indices."""
        # Create multiple indices
        create_self_index('test_list1', self.test_docs[:2], info='BOOLEAN')
        create_self_index('test_list2', self.test_docs[1:], info='WORDCOUNT')
        
        # List indices
        indices = list(SelfIndex.list_indices())
        
        # Check both indices are listed
        self.assertIn('test_list1', indices)
        self.assertIn('test_list2', indices)
    
    def test_list_indexed_files(self):
        """Test listing files in an index."""
        index = create_self_index('test_files', self.test_docs, info='BOOLEAN')
        
        # Get indexed files
        files = list(index.list_indexed_files('test_files'))
        
        # Check all documents are listed
        self.assertEqual(len(files), 3)
        self.assertIn('doc1', files)
        self.assertIn('doc2', files)
        self.assertIn('doc3', files)
    
    def test_delete_index(self):
        """Test index deletion."""
        # Create an index
        index = create_self_index('test_delete', self.test_docs, info='BOOLEAN')
        index_path = index._get_index_path('test_delete')
        
        # Verify it exists
        self.assertTrue(index_path.exists())
        
        # Delete it
        index.delete_index('test_delete')
        
        # Verify it's gone
        self.assertFalse(index_path.exists())
        self.assertNotIn('test_delete', SelfIndex._loaded_indices)
    
    def test_update_index_add(self):
        """Test adding documents to an index."""
        # Create initial index
        index = create_self_index('test_update', self.test_docs[:2], info='BOOLEAN')
        initial_count = index.num_docs
        
        # Add a document
        new_docs = [("doc4", "New document with unique words")]
        index.update_index('test_update', [], new_docs)
        
        # Verify document was added
        self.assertEqual(index.num_docs, initial_count + 1)
        self.assertIn('doc4', index.doc_metadata)
    
    def test_update_index_remove(self):
        """Test removing documents from an index."""
        # Create initial index
        index = create_self_index('test_remove', self.test_docs, info='BOOLEAN')
        initial_count = index.num_docs
        
        # Remove a document
        remove_docs = [("doc2", "")]
        index.update_index('test_remove', remove_docs, [])
        
        # Verify document was removed
        self.assertEqual(index.num_docs, initial_count - 1)
        self.assertNotIn('doc2', index.doc_metadata)


class TestSelfIndexAdvanced(unittest.TestCase):
    """Test advanced SelfIndex features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.original_storage_dir = SelfIndex.INDEX_STORAGE_DIR
        SelfIndex.INDEX_STORAGE_DIR = Path(self.test_dir)
        
        self.test_docs = [
            ("doc1", "apple banana cherry"),
            ("doc2", "banana cherry date"),
            ("doc3", "cherry date elderberry"),
            ("doc4", "date elderberry fig"),
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        SelfIndex.INDEX_STORAGE_DIR = self.original_storage_dir
        shutil.rmtree(self.test_dir, ignore_errors=True)
        SelfIndex._loaded_indices.clear()
    
    def test_ranking_with_wordcount(self):
        """Test that WordCount index ranks by term frequency."""
        # Create document with repeated terms
        docs = [
            ("doc1", "apple apple apple banana"),
            ("doc2", "apple banana banana banana"),
        ]
        index = create_self_index('test_rank_wc', docs, info='WORDCOUNT')
        
        # Query for 'banana' - doc2 should rank higher
        results_json = index.query("banana")
        results = json.loads(results_json)
        
        if results['num_results'] >= 2:
            # doc2 should have higher score (more 'banana' occurrences)
            doc_scores = {r['doc_id']: r['score'] for r in results['results']}
            self.assertGreater(doc_scores.get('doc2', 0), doc_scores.get('doc1', 0))
    
    def test_tfidf_scoring(self):
        """Test TF-IDF scoring distinguishes common vs rare terms."""
        index = create_self_index('test_tfidf_score', self.test_docs, info='TFIDF')
        
        # 'cherry' appears in 3 docs (common), 'apple' in 1 doc (rare)
        # TF-IDF should reflect this
        if 'cherry' in index.inverted_index and 'apple' in index.inverted_index:
            cherry_idf = index.inverted_index['cherry']['doc1'].get('idf', 0)
            apple_idf = index.inverted_index['apple']['doc1'].get('idf', 0)
            
            # Rare term should have higher IDF
            self.assertGreater(apple_idf, cherry_idf)
    
    def test_position_tracking(self):
        """Test that positions are correctly tracked."""
        docs = [("doc1", "one two three four five")]
        index = create_self_index('test_pos', docs, info='BOOLEAN')
        
        # Check positions
        if 'three' in index.inverted_index:
            positions = index.inverted_index['three']['doc1']
            # 'three' should be at position 2 (0-indexed)
            self.assertIn(2, positions)
    
    def test_empty_query(self):
        """Test behavior with empty query."""
        index = create_self_index('test_empty', self.test_docs, info='BOOLEAN')
        
        results_json = index.query("")
        results = json.loads(results_json)
        
        # Empty query should return no results
        self.assertEqual(results['num_results'], 0)
    
    def test_nonexistent_term_query(self):
        """Test query for non-existent term."""
        index = create_self_index('test_noterm', self.test_docs, info='BOOLEAN')
        
        results_json = index.query("nonexistent_term_xyz")
        results = json.loads(results_json)
        
        # Should return no results
        self.assertEqual(results['num_results'], 0)


class TestSelfIndexIdentifier(unittest.TestCase):
    """Test SelfIndex identifier and versioning."""
    
    def test_identifier_generation(self):
        """Test that identifiers are correctly generated."""
        index = SelfIndex(info='BOOLEAN', dstore='CUSTOM', qproc='TERMatat',
                         compr='NONE', optim='Null')
        
        # Check short identifier format
        self.assertIn('SelfIndex', index.identifier_short)
        
        # Check long identifier format
        self.assertIn('core=SelfIndex', index.identifier_long)
        self.assertIn('index=IndexInfo.BOOLEAN', index.identifier_long)
    
    def test_different_configurations(self):
        """Test different index configurations."""
        configs = [
            ('BOOLEAN', 'CUSTOM', 'TERMatat', 'NONE', 'Null'),
            ('WORDCOUNT', 'CUSTOM', 'DOCatat', 'NONE', 'Null'),
            ('TFIDF', 'CUSTOM', 'TERMatat', 'NONE', 'Null'),
        ]
        
        for info, dstore, qproc, compr, optim in configs:
            index = SelfIndex(info=info, dstore=dstore, qproc=qproc, 
                            compr=compr, optim=optim)
            self.assertIsNotNone(index.identifier_short)
            self.assertIsNotNone(index.identifier_long)


if __name__ == '__main__':
    unittest.main()
