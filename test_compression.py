"""
Unit tests for compression utilities.
"""

import unittest
from compression import (
    NoCompression, CustomCompression, LibraryCompression,
    get_compressor
)


class TestCustomCompression(unittest.TestCase):
    """Test custom compression methods."""
    
    def setUp(self):
        self.compressor = CustomCompression()
    
    def test_varbyte_encoding_small_numbers(self):
        """Test variable-byte encoding with small numbers."""
        numbers = [1, 2, 3, 127]
        encoded = self.compressor._encode_varbyte(numbers)
        decoded = self.compressor._decode_varbyte(encoded)
        self.assertEqual(decoded, numbers)
    
    def test_varbyte_encoding_large_numbers(self):
        """Test variable-byte encoding with large numbers."""
        numbers = [128, 256, 1000, 10000]
        encoded = self.compressor._encode_varbyte(numbers)
        decoded = self.compressor._decode_varbyte(encoded)
        self.assertEqual(decoded, numbers)
    
    def test_varbyte_encoding_mixed(self):
        """Test variable-byte encoding with mixed numbers."""
        numbers = [0, 1, 127, 128, 255, 256, 1000]
        encoded = self.compressor._encode_varbyte(numbers)
        decoded = self.compressor._decode_varbyte(encoded)
        self.assertEqual(decoded, numbers)
    
    def test_gap_encoding(self):
        """Test gap encoding."""
        positions = [0, 5, 10, 15, 100]
        gaps = self.compressor._gap_encode(positions)
        self.assertEqual(gaps, [0, 5, 5, 5, 85])
        
        # Decode back
        decoded = self.compressor._gap_decode(gaps)
        self.assertEqual(decoded, positions)
    
    def test_compress_posting_list(self):
        """Test compressing a posting list."""
        positions = [0, 1, 2, 5, 10, 50, 100]
        compressed = self.compressor.compress_posting_list(positions)
        decompressed = self.compressor.decompress_posting_list(compressed)
        self.assertEqual(decompressed, positions)
    
    def test_compress_empty_list(self):
        """Test compressing an empty list."""
        compressed = self.compressor.compress_posting_list([])
        decompressed = self.compressor.decompress_posting_list(compressed)
        self.assertEqual(decompressed, [])
    
    def test_compress_postings_with_metadata(self):
        """Test compressing postings with metadata."""
        postings = {
            'positions': [0, 5, 10],
            'tf': 3,
            'tfidf': 1.5
        }
        compressed = self.compressor.compress_postings(postings)
        decompressed = self.compressor.decompress_postings(compressed)
        
        self.assertEqual(decompressed['positions'], postings['positions'])
        self.assertEqual(decompressed['tf'], postings['tf'])
        self.assertEqual(decompressed['tfidf'], postings['tfidf'])
    
    def test_compression_ratio(self):
        """Test that compression actually reduces size."""
        # Large list of sequential positions
        positions = list(range(1000))
        
        # Uncompressed size (rough estimate)
        import pickle
        uncompressed_size = len(pickle.dumps(positions))
        
        # Compressed size
        compressed = self.compressor.compress_posting_list(positions)
        compressed_size = len(compressed)
        
        # Compression should reduce size
        self.assertLess(compressed_size, uncompressed_size)


class TestLibraryCompression(unittest.TestCase):
    """Test library-based compression."""
    
    def setUp(self):
        self.compressor = LibraryCompression()
    
    def test_compress_posting_list(self):
        """Test compressing a posting list with zlib."""
        positions = [0, 1, 2, 5, 10, 50, 100]
        compressed = self.compressor.compress_posting_list(positions)
        decompressed = self.compressor.decompress_posting_list(compressed)
        self.assertEqual(decompressed, positions)
    
    def test_compress_empty_list(self):
        """Test compressing an empty list."""
        compressed = self.compressor.compress_posting_list([])
        decompressed = self.compressor.decompress_posting_list(compressed)
        self.assertEqual(decompressed, [])
    
    def test_compress_postings_with_metadata(self):
        """Test compressing postings with metadata."""
        postings = {
            'positions': [0, 5, 10],
            'tf': 3,
            'tfidf': 1.5
        }
        compressed = self.compressor.compress_postings(postings)
        decompressed = self.compressor.decompress_postings(compressed)
        
        self.assertEqual(decompressed['positions'], postings['positions'])
        self.assertEqual(decompressed['tf'], postings['tf'])
        self.assertEqual(decompressed['tfidf'], postings['tfidf'])
    
    def test_compression_levels(self):
        """Test different compression levels."""
        positions = list(range(1000))
        
        # Test different levels
        for level in [1, 6, 9]:
            compressor = LibraryCompression(compression_level=level)
            compressed = compressor.compress_posting_list(positions)
            decompressed = compressor.decompress_posting_list(compressed)
            self.assertEqual(decompressed, positions)
    
    def test_compression_ratio(self):
        """Test that zlib compression reduces size."""
        # Large list of positions
        positions = list(range(1000))
        
        # Uncompressed size
        import json
        uncompressed_size = len(json.dumps(positions).encode('utf-8'))
        
        # Compressed size
        compressed = self.compressor.compress_posting_list(positions)
        compressed_size = len(compressed)
        
        # Compression should reduce size
        self.assertLess(compressed_size, uncompressed_size)


class TestNoCompression(unittest.TestCase):
    """Test no compression (identity function)."""
    
    def setUp(self):
        self.compressor = NoCompression()
    
    def test_compress_posting_list(self):
        """Test identity compression."""
        positions = [0, 1, 2, 5, 10]
        compressed = self.compressor.compress_posting_list(positions)
        decompressed = self.compressor.decompress_posting_list(compressed)
        self.assertEqual(decompressed, positions)


class TestCompressionFactory(unittest.TestCase):
    """Test compression factory function."""
    
    def test_get_compressor_none(self):
        """Test getting no compression."""
        compressor = get_compressor('NONE')
        self.assertIsInstance(compressor, NoCompression)
    
    def test_get_compressor_custom(self):
        """Test getting custom compression."""
        compressor = get_compressor('CODE')
        self.assertIsInstance(compressor, CustomCompression)
    
    def test_get_compressor_library(self):
        """Test getting library compression."""
        compressor = get_compressor('CLIB')
        self.assertIsInstance(compressor, LibraryCompression)
    
    def test_get_compressor_invalid(self):
        """Test invalid compression type."""
        with self.assertRaises(ValueError):
            get_compressor('INVALID')


class TestCompressionComparison(unittest.TestCase):
    """Compare different compression methods."""
    
    def test_all_methods_produce_same_result(self):
        """Test that all compression methods produce the same logical result."""
        positions = [0, 5, 10, 15, 20, 100, 200, 500]
        
        compressors = [
            NoCompression(),
            CustomCompression(),
            LibraryCompression()
        ]
        
        for compressor in compressors:
            compressed = compressor.compress_posting_list(positions)
            decompressed = compressor.decompress_posting_list(compressed)
            self.assertEqual(decompressed, positions, 
                           f"Failed for {compressor.__class__.__name__}")
    
    def test_compression_ratios(self):
        """Compare compression ratios of different methods."""
        # Create a large list of sequential positions
        positions = list(range(1000))
        
        import pickle
        none_compressor = NoCompression()
        custom_compressor = CustomCompression()
        lib_compressor = LibraryCompression()
        
        none_size = len(none_compressor.compress_posting_list(positions))
        custom_size = len(custom_compressor.compress_posting_list(positions))
        lib_size = len(lib_compressor.compress_posting_list(positions))
        
        # Both compression methods should reduce size compared to no compression
        self.assertLess(custom_size, none_size)
        self.assertLess(lib_size, none_size)
        
        print(f"\nCompression comparison for 1000 sequential positions:")
        print(f"  No compression: {none_size} bytes")
        print(f"  Custom compression: {custom_size} bytes ({custom_size/none_size*100:.1f}%)")
        print(f"  Library compression: {lib_size} bytes ({lib_size/none_size*100:.1f}%)")


if __name__ == '__main__':
    unittest.main(verbosity=2)
