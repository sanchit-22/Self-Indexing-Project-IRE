"""
Compression utilities for posting lists.

Implements two compression methods:
1. Custom compression (variable-byte encoding and gap encoding)
2. Library-based compression (zlib)
"""

import struct
from typing import List, Dict, Any
import zlib
import json


class CompressionBase:
    """Base class for compression methods."""
    
    def compress_posting_list(self, posting_list: List[int]) -> bytes:
        """Compress a list of positions."""
        raise NotImplementedError
    
    def decompress_posting_list(self, compressed_data: bytes) -> List[int]:
        """Decompress back to list of positions."""
        raise NotImplementedError
    
    def compress_postings(self, postings: Dict[str, Any]) -> bytes:
        """Compress entire posting structure."""
        raise NotImplementedError
    
    def decompress_postings(self, compressed_data: bytes) -> Dict[str, Any]:
        """Decompress entire posting structure."""
        raise NotImplementedError


class NoCompression(CompressionBase):
    """No compression - identity function."""
    
    def compress_posting_list(self, posting_list: List[int]) -> bytes:
        """Return positions as-is (pickled)."""
        import pickle
        return pickle.dumps(posting_list)
    
    def decompress_posting_list(self, compressed_data: bytes) -> List[int]:
        """Return positions as-is (unpickled)."""
        import pickle
        return pickle.loads(compressed_data)
    
    def compress_postings(self, postings: Dict[str, Any]) -> bytes:
        """Return postings as-is (pickled)."""
        import pickle
        return pickle.dumps(postings)
    
    def decompress_postings(self, compressed_data: bytes) -> Dict[str, Any]:
        """Return postings as-is (unpickled)."""
        import pickle
        return pickle.loads(compressed_data)


class CustomCompression(CompressionBase):
    """
    Custom compression using variable-byte encoding and gap encoding.
    
    Gap encoding: Store differences between consecutive positions instead of absolute positions.
    Variable-byte encoding: Use variable number of bytes to represent integers.
    """
    
    def _encode_varbyte(self, numbers: List[int]) -> bytes:
        """
        Encode list of integers using variable-byte encoding.
        
        Args:
            numbers: List of non-negative integers
            
        Returns:
            Encoded bytes
        """
        result = bytearray()
        for num in numbers:
            # Encode each number using 7 bits per byte, with continuation bit
            bytes_list = []
            while num >= 128:
                bytes_list.append((num & 0x7F) | 0x80)
                num >>= 7
            bytes_list.append(num & 0x7F)
            result.extend(bytes_list)
        return bytes(result)
    
    def _decode_varbyte(self, data: bytes) -> List[int]:
        """
        Decode variable-byte encoded data.
        
        Args:
            data: Encoded bytes
            
        Returns:
            List of decoded integers
        """
        numbers = []
        current = 0
        shift = 0
        
        for byte in data:
            current |= (byte & 0x7F) << shift
            shift += 7
            
            if (byte & 0x80) == 0:
                # End of current number
                numbers.append(current)
                current = 0
                shift = 0
        
        return numbers
    
    def _gap_encode(self, positions: List[int]) -> List[int]:
        """
        Convert absolute positions to gaps.
        
        Args:
            positions: Sorted list of positions
            
        Returns:
            List of gaps
        """
        if not positions:
            return []
        
        gaps = [positions[0]]
        for i in range(1, len(positions)):
            gaps.append(positions[i] - positions[i-1])
        return gaps
    
    def _gap_decode(self, gaps: List[int]) -> List[int]:
        """
        Convert gaps back to absolute positions.
        
        Args:
            gaps: List of gaps
            
        Returns:
            List of absolute positions
        """
        if not gaps:
            return []
        
        positions = [gaps[0]]
        for i in range(1, len(gaps)):
            positions.append(positions[-1] + gaps[i])
        return positions
    
    def compress_posting_list(self, posting_list: List[int]) -> bytes:
        """
        Compress a list of positions using gap encoding and variable-byte encoding.
        
        Args:
            posting_list: Sorted list of positions
            
        Returns:
            Compressed bytes
        """
        if not posting_list:
            return b''
        
        # Sort positions (should already be sorted)
        sorted_positions = sorted(posting_list)
        
        # Apply gap encoding
        gaps = self._gap_encode(sorted_positions)
        
        # Apply variable-byte encoding
        compressed = self._encode_varbyte(gaps)
        
        return compressed
    
    def decompress_posting_list(self, compressed_data: bytes) -> List[int]:
        """
        Decompress a posting list.
        
        Args:
            compressed_data: Compressed bytes
            
        Returns:
            List of positions
        """
        if not compressed_data:
            return []
        
        # Decode variable-byte
        gaps = self._decode_varbyte(compressed_data)
        
        # Decode gaps
        positions = self._gap_decode(gaps)
        
        return positions
    
    def compress_postings(self, postings: Dict[str, Any]) -> bytes:
        """
        Compress entire posting structure.
        
        Args:
            postings: Dictionary with posting data
            
        Returns:
            Compressed bytes
        """
        # Handle different posting structures
        if isinstance(postings, list):
            # Simple list of positions
            return self.compress_posting_list(postings)
        elif isinstance(postings, dict):
            # Complex structure with tf, tfidf, etc.
            if 'positions' in postings:
                # Compress positions separately
                compressed_positions = self.compress_posting_list(postings['positions'])
                # Store other fields as JSON
                other_data = {k: v for k, v in postings.items() if k != 'positions'}
                other_json = json.dumps(other_data).encode('utf-8')
                
                # Combine: [length of positions][compressed positions][other data]
                result = struct.pack('I', len(compressed_positions))
                result += compressed_positions
                result += other_json
                return result
            else:
                # No positions, just serialize as JSON
                import pickle
                return pickle.dumps(postings)
        else:
            import pickle
            return pickle.dumps(postings)
    
    def decompress_postings(self, compressed_data: bytes) -> Dict[str, Any]:
        """
        Decompress entire posting structure.
        
        Args:
            compressed_data: Compressed bytes
            
        Returns:
            Posting dictionary
        """
        if not compressed_data:
            return {}
        
        try:
            # Try to extract length header
            if len(compressed_data) >= 4:
                positions_length = struct.unpack('I', compressed_data[:4])[0]
                if positions_length <= len(compressed_data) - 4:
                    # Extract compressed positions
                    compressed_positions = compressed_data[4:4+positions_length]
                    positions = self.decompress_posting_list(compressed_positions)
                    
                    # Extract other data
                    other_json = compressed_data[4+positions_length:]
                    if other_json:
                        other_data = json.loads(other_json.decode('utf-8'))
                        other_data['positions'] = positions
                        return other_data
                    else:
                        return {'positions': positions}
        except:
            pass
        
        # Fall back to pickle
        import pickle
        return pickle.loads(compressed_data)


class LibraryCompression(CompressionBase):
    """
    Library-based compression using zlib.
    """
    
    def __init__(self, compression_level: int = 6):
        """
        Initialize library compression.
        
        Args:
            compression_level: Compression level (0-9, default 6)
        """
        self.compression_level = compression_level
    
    def compress_posting_list(self, posting_list: List[int]) -> bytes:
        """
        Compress a list of positions using zlib.
        
        Args:
            posting_list: List of positions
            
        Returns:
            Compressed bytes
        """
        # Serialize as JSON
        data = json.dumps(posting_list).encode('utf-8')
        
        # Compress with zlib
        compressed = zlib.compress(data, level=self.compression_level)
        
        return compressed
    
    def decompress_posting_list(self, compressed_data: bytes) -> List[int]:
        """
        Decompress a posting list.
        
        Args:
            compressed_data: Compressed bytes
            
        Returns:
            List of positions
        """
        if not compressed_data:
            return []
        
        # Decompress with zlib
        data = zlib.decompress(compressed_data)
        
        # Deserialize JSON
        positions = json.loads(data.decode('utf-8'))
        
        return positions
    
    def compress_postings(self, postings: Dict[str, Any]) -> bytes:
        """
        Compress entire posting structure.
        
        Args:
            postings: Posting dictionary
            
        Returns:
            Compressed bytes
        """
        # Serialize as JSON
        data = json.dumps(postings).encode('utf-8')
        
        # Compress with zlib
        compressed = zlib.compress(data, level=self.compression_level)
        
        return compressed
    
    def decompress_postings(self, compressed_data: bytes) -> Dict[str, Any]:
        """
        Decompress entire posting structure.
        
        Args:
            compressed_data: Compressed bytes
            
        Returns:
            Posting dictionary
        """
        if not compressed_data:
            return {}
        
        # Decompress with zlib
        data = zlib.decompress(compressed_data)
        
        # Deserialize JSON
        postings = json.loads(data.decode('utf-8'))
        
        return postings


def get_compressor(compression_type: str) -> CompressionBase:
    """
    Get a compressor instance based on type.
    
    Args:
        compression_type: 'NONE', 'CODE', or 'CLIB'
        
    Returns:
        Compressor instance
    """
    if compression_type == 'NONE':
        return NoCompression()
    elif compression_type == 'CODE':
        return CustomCompression()
    elif compression_type == 'CLIB':
        return LibraryCompression()
    else:
        raise ValueError(f"Unknown compression type: {compression_type}")
