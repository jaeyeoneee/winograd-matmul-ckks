import numpy as np
from typing import List, Tuple, Optional, Union
import time
import logging

from desilofhe import Engine
    
class CKKSWrapper:
    """
    Wrapper for CKKS homomorphic encryption operations using liberate-fhe
    Provides simplified interface for basic operations needed for Winograd acceleration
    """
    
    def __init__(self, 
                poly_modulus_degree: int = 13,
                special_prime_count: int = 1,
                mode: str = "cpu",
                thread_count: int = 256,
                ):
        """
        Initialize CKKS context with parameters suitable for Winograd operations
        
        Args:
            poly_modulus_degree: Polynomial modulus degree (N)
            coeff_modulus_bits: Bit sizes for modulus chain
            scale_bits: Scale factor bit size
        """
        
        self.poly_modulus_degree = poly_modulus_degree
        self.mode = mode
        self.thread_count = thread_count
        self.special_prime_count = special_prime_count
        
        self._setup_context()
        self._generate_keys()
        
        # Operation counters for analysis
        self.stats = {
            'ct_ct_multiplications': 0,
            'rotations': 0,
            'rescalings': 0,
            'encryptions': 0,
            'decryptions': 0
        }
        
        print(f"CKKS Context initialized:")
        print(f"  Polynomial degree: {self.poly_modulus_degree}")
        print(f"  SIMD slots: {self.num_slots}")
        print(f"  Levels: {self.max_level + 1}")
        # print(f"  Scale: 2^{scale_bits}")
    
    
    def _setup_context(self):
      self.engine = Engine(log_coeff_count=self.poly_modulus_degree, special_prime_count= self.special_prime_count, mode=self.mode, thread_count=self.thread_count)
      self.max_level = self.engine.max_level
      self.num_slots = self.engine.slot_count
      
    
    def _generate_keys(self):
      self.secret_key = self.engine.create_secret_key()
      self.public_key = self.engine.create_public_key(self.secret_key)
      self.relin_key = self.engine.create_relinearization_key(self.secret_key)
      self.rotate_key = self.engine.create_rotation_key(self.secret_key)
      
    
    def encode_and_encrypt(self, data: Union[List, np.ndarray], 
                          level: Optional[int] = None):
        """
        Encode and encrypt data into a CKKS ciphertext
        
        Args:
            data: Input data (real numbers)
            level: Target level (None for max level)
        Returns:
            Encrypted ciphertext
        """
        if isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        
        if len(data) > self.num_slots:
            data = data[:self.num_slots]
            print(f"Warning: Data truncated to {self.num_slots} slots")
        elif len(data) < self.num_slots:
            padded = np.zeros(self.num_slots)
            padded[:len(data)] = data
            data = padded
        
        target_level = level if level is not None else self.max_level
        
        ciphertext = self.engine.encrypt(data, self.public_key, level=target_level)
        
        self.stats['encryptions'] += 1
        return ciphertext
    
    
    def decrypt_and_decode(self, ciphertext, 
                          expected_size: Optional[int] = None) -> np.ndarray:
        """
        Decrypt and decode a CKKS ciphertext
        
        Args:
            ciphertext: Input ciphertext
            expected_size: Expected output size (for truncation)
        Returns:
            Decrypted data
        """

        data = self.engine.decrypt(ciphertext, self.secret_key)
        
        if expected_size is not None and len(data) > expected_size:
            data = data[:expected_size]
        
        self.stats['decryptions'] += 1
        return data
    
    
    def multiply_ciphertexts(self, ct1, ct2):
        """
        Multiply two ciphertexts (ct-ct multiplication)
        
        Args:
            ct1, ct2: Input ciphertexts
            auto_rescale: Whether to automatically rescale result
        Returns:
            Product ciphertext
        """
        
        result = self.engine.multiply(ct1, ct2, self.relin_key)
        
        self.stats['ct_ct_multiplications'] += 1
        return result
    
    
    def multiply_plaintext(self, ciphertext, 
                          plaintext_data: Union[float, np.ndarray]):
        """
        Multiply ciphertext by plaintext (ct-pt multiplication)
        
        Args:
            ciphertext: Input ciphertext
            plaintext_data: Plaintext data (scalar or array)
        Returns:
            Product ciphertext
        """
        if isinstance(plaintext_data, (int, float)):
            plaintext_data = np.full(self.num_slots, plaintext_data)
        elif len(plaintext_data) < self.num_slots:
            # Pad plaintext
            padded = np.zeros(self.num_slots)
            padded[:len(plaintext_data)] = plaintext_data
            plaintext_data = padded

        result = self.engine.multiply(ciphertext, plaintext_data)
        
        return result
    
    def add_ciphertexts(self, ct1, ct2):

        result = self.engine.add(ct1, ct2)
                
        return result
    
    def rotate_slots(self, ciphertext, steps: int):
        """
        Rotate SIMD slots by given number of steps
        
        Args:
            ciphertext: Input ciphertext
            steps: Number of positions to rotate (positive = left, negative = right)
        Returns:
            Rotated ciphertext
        """

        result = self.engine.rotate(ciphertext, self.rotate_key, steps)
        
        self.stats['rotations'] += 1
        return result
    
    def rescale(self, ciphertext):
        """
        Rescale ciphertext to reduce scale and consume one level
        
        Args:
            ciphertext: Input ciphertext
        Returns:
            Rescaled ciphertext
        """
        if ciphertext.level == 0:
            print("Warning: Cannot rescale at level 0")
            return ciphertext
        
        result = self.engine.rescale(ciphertext)
        
        self.stats['rescalings'] += 1
        return result
    
    def get_ciphertext_info(self, ciphertext) -> dict:
        """Get information about a ciphertext"""
        return {
            'level': ciphertext.level,
            'scale': ciphertext.scale,
            'size': ciphertext.size,
            'remaining_levels': ciphertext.level
        }
    
    def pack_matrix_diagonals(self, matrix: np.ndarray, 
                             num_diagonals: int):
        """
        Pack matrix diagonals into ciphertexts for diagonal method
        
        Args:
            matrix: Input matrix (rows x cols)
            num_diagonals: Number of diagonals to pack
        Returns:
            List of ciphertexts, one per diagonal
        """
        rows, cols = matrix.shape
        diagonals = []
        
        for k in range(num_diagonals):
            # Extract k-th diagonal
            diagonal_data = np.zeros(self.num_slots)
            diagonal_idx = 0
            
            for i in range(rows):
                j = (i + k) % cols  # Circular wrapping
                if diagonal_idx < self.num_slots:
                    diagonal_data[diagonal_idx] = matrix[i, j]
                    diagonal_idx += 1
            
            # Encrypt diagonal
            diagonal_ct = self.encode_and_encrypt(diagonal_data)
            diagonals.append(diagonal_ct)
        
        return diagonals
    
    def pack_winograd_tiles(self, matrix: np.ndarray, 
                           tile_m: int, tile_r: int):
        """
        Pack matrix into Winograd-compatible tiles
        
        Args:
            matrix: Input matrix
            tile_m: Tile height
            tile_r: Tile width
        Returns:
            List of encrypted tiles
        """
        rows, cols = matrix.shape
        tiles = []
        
        for i in range(0, rows, tile_m):
            for j in range(0, cols, tile_r):
                # Extract tile
                tile_end_i = min(i + tile_m, rows)
                tile_end_j = min(j + tile_r, cols)
                tile = matrix[i:tile_end_i, j:tile_end_j]
                
                # Pad if necessary
                if tile.shape != (tile_m, tile_r):
                    padded_tile = np.zeros((tile_m, tile_r))
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile
                
                # Flatten and encrypt
                tile_flat = tile.flatten()
                tile_ct = self.encode_and_encrypt(tile_flat)
                tiles.append(tile_ct)
        
        return tiles
    
    def reset_stats(self):
        for key in self.stats:
            self.stats[key] = 0
    
    def print_stats(self):
        print("\n=== CKKS Operation Statistics ===")
        for op, count in self.stats.items():
            print(f"{op}: {count}")

# Test and demonstration functions
def test_ckks_wrapper():
    """Test basic CKKS wrapper functionality"""
    print("=== Testing CKKS Wrapper ===\n")
    
    # Initialize CKKS wrapper
    ckks = CKKSWrapper(
        poly_modulus_degree=13
    )
    
    print(f"\n1. Basic encryption/decryption test:")
    
    # Test data
    test_data = [1.5, 2.7, 3.14, -1.2, 0.5]
    print(f"Original data: {test_data}")
    
    # Encrypt
    ciphertext = ckks.encode_and_encrypt(test_data)
    print(f"Encrypted (level {ciphertext.level})")
    
    # Decrypt
    decrypted = ckks.decrypt_and_decode(ciphertext, len(test_data))
    print(f"Decrypted: {decrypted}")
    print(f"Error: {np.max(np.abs(np.array(test_data) - decrypted)):.2e}")
    
    print(f"\n2. Ciphertext-ciphertext multiplication:")
    
    # Create two ciphertexts
    data1 = [2.0, 3.0, 4.0]
    data2 = [1.5, 2.0, 0.5]
    
    ct1 = ckks.encode_and_encrypt(data1)
    ct2 = ckks.encode_and_encrypt(data2)
    
    print(f"Data1: {data1}")
    print(f"Data2: {data2}")
    
    # Multiply
    ct_product = ckks.multiply_ciphertexts(ct1, ct2)
    result = ckks.decrypt_and_decode(ct_product, len(data1))
    expected = np.array(data1) * np.array(data2)
    
    print(f"Result: {result}")
    print(f"Expected: {expected}")
    print(f"Error: {np.max(np.abs(expected - result)):.2e}")
    
    print(f"\n3. Rotation test:")
    
    # Test rotation
    data = [1, 2, 3, 4, 5, 0, 0, 0]  # Pad with zeros
    ct = ckks.encode_and_encrypt(data)
    
    rotated_ct = ckks.rotate_slots(ct, 2)  # Rotate left by 2
    rotated_result = ckks.decrypt_and_decode(rotated_ct, len(data))
    
    print(f"Original: {data}")
    print(f"Rotated by 2: {rotated_result}")
    
    print(f"\n4. Matrix diagonal packing test:")
    
    # Test matrix
    matrix = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=np.float64)
    
    print(f"Matrix:\n{matrix}")
    
    diagonals = ckks.pack_matrix_diagonals(matrix, num_diagonals=3)
    print(f"Packed into {len(diagonals)} diagonal ciphertexts")
    
    # Verify first diagonal
    first_diagonal = ckks.decrypt_and_decode(diagonals[0], 3)
    second_diagonal = ckks.decrypt_and_decode(diagonals[1], 3)
    third_diagonal = ckks.decrypt_and_decode(diagonals[2], 3)
    print(f"First diagonal: {first_diagonal} (should be [1, 5, 9])")
    print(f"Second diagonal: {second_diagonal} (should be [2, 6, 7])")
    print(f"Third diagonal: {third_diagonal} (should be [3, 4, 8])")
    
    # Print final stats
    ckks.print_stats()
    
    return ckks

def benchmark_operations(ckks: CKKSWrapper, num_operations: int = 100):
    """Benchmark basic operations"""
    print(f"\n=== Benchmarking {num_operations} operations ===")
    
    # Prepare test data
    data1 = np.random.randn(1000)
    data2 = np.random.randn(1000)
    
    ct1 = ckks.encode_and_encrypt(data1)
    ct2 = ckks.encode_and_encrypt(data2)
    
    ckks.reset_stats()
    
    # Benchmark ct-ct multiplication
    start_time = time.time()
    for _ in range(num_operations):
        result = ckks.multiply_ciphertexts(ct1, ct2)
    mult_time = time.time() - start_time
    
    # Benchmark rotation
    start_time = time.time()
    for i in range(num_operations):
        result = ckks.rotate_slots(ct1, i % 10)
    rotation_time = time.time() - start_time
    
    print(f"Multiplication time: {mult_time/num_operations*1000:.2f} ms per operation")
    print(f"Rotation time: {rotation_time/num_operations*1000:.2f} ms per operation")
    
    ckks.print_stats()

# Run tests if executed directly
if __name__ == "__main__":
    ckks_wrapper = test_ckks_wrapper()
    
    # Run benchmark
    benchmark_operations(ckks_wrapper, num_operations=10)  # Smaller number for demo