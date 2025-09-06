import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import time
from abc import ABC, abstractmethod

@dataclass
class WinogradConfig:
    """Winograd configuration parameters"""
    m: int  # output tile height
    r: int  # filter width (number of consecutive diagonals)
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"F({self.m},{self.r})"
    
    @property
    def reduction_factor(self) -> float:
        """Multiplication reduction factor"""
        return 1 - (self.m + self.r - 1) / (self.m * self.r)
    
    @property
    def multiplications_per_tile(self) -> int:
        """Number of ct-ct multiplications per tile"""
        return self.m + self.r - 1

@dataclass
class CKKSParams:
    """CKKS scheme parameters"""
    N: int = 2**16  # polynomial degree
    log_Q: int = 1800  # ciphertext modulus bit length
    levels: int = 30  # multiplicative depth
    scale_bits: int = 60  # scaling factor bits
    
    @property
    def simd_slots(self) -> int:
        """Number of SIMD slots available"""
        return self.N // 2

class SIMDSlotLayout:
    """
    Implements the hierarchical SIMD slot layout from Section 3.4
    slot(a, τ, ℓ) = a · (T*m) + τ · m + ℓ
    """
    
    def __init__(self, 
                 h: int,  # number of attention heads
                 L: int,  # sequence length
                 d_model: int,  # model dimension
                 winograd_config: WinogradConfig,
                 ckks_params: CKKSParams):
        
        self.h = h
        self.L = L
        self.d_model = d_model
        self.wg_config = winograd_config
        self.ckks_params = ckks_params
        
        # Calculate derived parameters
        self.d_k = d_model // h  # dimension per head
        self.T = self._calculate_tiles_per_head()
        self.elements_per_head = self.T * winograd_config.m
        self.total_elements = h * self.elements_per_head
        
        # Validate configuration
        self._validate_configuration()
        
    def _calculate_tiles_per_head(self) -> int:
        """Calculate number of tiles per attention head"""
        tiles_rows = (self.L + self.wg_config.m - 1) // self.wg_config.m
        tiles_cols = (self.d_k + self.wg_config.r - 1) // self.wg_config.r
        return tiles_rows * tiles_cols
    
    def _validate_configuration(self):
        """Validate that the configuration fits in SIMD slots"""
        available_slots = self.ckks_params.simd_slots
        if self.total_elements > available_slots:
            raise ValueError(f"Configuration requires {self.total_elements} slots "
                           f"but only {available_slots} available")
    
    def slot_index(self, head: int, tile: int, element: int) -> int:
        """
        Compute slot index using equation (15) from the paper:
        slot(a, τ, ℓ) = a · (T*m) + τ · m + ℓ
        """
        if not (0 <= head < self.h):
            raise ValueError(f"Head index {head} out of range [0, {self.h})")
        if not (0 <= tile < self.T):
            raise ValueError(f"Tile index {tile} out of range [0, {self.T})")
        if not (0 <= element < self.wg_config.m):
            raise ValueError(f"Element index {element} out of range [0, {self.wg_config.m})")
            
        return head * self.elements_per_head + tile * self.wg_config.m + element
    
    def reverse_slot_index(self, slot_idx: int) -> Tuple[int, int, int]:
        """Reverse the slot index to get (head, tile, element)"""
        if not (0 <= slot_idx < self.total_elements):
            raise ValueError(f"Slot index {slot_idx} out of range")
            
        head = slot_idx // self.elements_per_head
        remainder = slot_idx % self.elements_per_head
        tile = remainder // self.wg_config.m
        element = remainder % self.wg_config.m
        
        return head, tile, element
    
    def get_head_slot_range(self, head: int) -> Tuple[int, int]:
        """Get the slot range for a specific head"""
        start = head * self.elements_per_head
        end = start + self.elements_per_head
        return start, end
    
    def get_tile_slot_range(self, head: int, tile: int) -> Tuple[int, int]:
        """Get the slot range for a specific tile within a head"""
        start = self.slot_index(head, tile, 0)
        end = start + self.wg_config.m
        return start, end
    
    def calculate_packing_efficiency(self) -> float:
        """Calculate slot utilization efficiency (equation 101)"""
        return self.total_elements / self.ckks_params.simd_slots
    
    def visualize_layout(self, max_display_slots: int = 128) -> None:
        """Visualize the slot layout"""
        display_slots = min(max_display_slots, self.total_elements)
        
        # Create visualization data
        head_colors = []
        tile_patterns = []
        
        for slot in range(display_slots):
            head, tile, element = self.reverse_slot_index(slot)
            head_colors.append(head)
            tile_patterns.append(tile % 4)  # Use modulo for pattern variety
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Head-based coloring
        ax1.imshow([head_colors], aspect='auto', cmap='tab10')
        ax1.set_title(f'SIMD Slot Layout - Head Assignment (showing first {display_slots} slots)')
        ax1.set_xlabel('Slot Index')
        ax1.set_ylabel('Heads')
        ax1.set_yticks([])
        
        # Add head boundaries
        for h in range(1, self.h):
            boundary = h * self.elements_per_head
            if boundary < display_slots:
                ax1.axvline(x=boundary-0.5, color='white', linewidth=2)
        
        # Tile-based patterns
        ax2.imshow([tile_patterns], aspect='auto', cmap='Pastel1')
        ax2.set_title('Tile Patterns within Heads')
        ax2.set_xlabel('Slot Index')
        ax2.set_ylabel('Tiles')
        ax2.set_yticks([])
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"Layout Statistics:")
        print(f"- Total heads: {self.h}")
        print(f"- Elements per head: {self.elements_per_head}")
        print(f"- Total slots used: {self.total_elements}")
        print(f"- Available SIMD slots: {self.ckks_params.simd_slots}")
        print(f"- Packing efficiency: {self.calculate_packing_efficiency():.2%}")

class MultiHeadWinogradProcessor:
    """
    Implements multi-head parallel processing using the SIMD layout
    """
    
    def __init__(self, layout: SIMDSlotLayout):
        self.layout = layout
        self.cached_transforms = {}
        self.rotation_patterns = self._compute_rotation_patterns()
        
    def _compute_rotation_patterns(self) -> Dict[str, List[int]]:
        """Compute rotation patterns for different operations"""
        patterns = {
            'gather': list(range(self.layout.wg_config.r)),
            'tile_boundary': [self.layout.wg_config.m, -self.layout.wg_config.m],
            'head_boundary': [self.layout.elements_per_head, -self.layout.elements_per_head]
        }
        return patterns
    
    def pack_multi_head_data(self, 
                           queries: np.ndarray, 
                           keys: np.ndarray, 
                           values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Pack Q, K, V matrices for multiple heads into SIMD slots
        
        Args:
            queries: [h, L, d_k] - queries for each head
            keys: [h, L, d_k] - keys for each head  
            values: [h, L, d_k] - values for each head
            
        Returns:
            Dictionary with packed arrays
        """
        if queries.shape != (self.layout.h, self.layout.L, self.layout.d_k):
            raise ValueError(f"Expected queries shape {(self.layout.h, self.layout.L, self.layout.d_k)}, "
                           f"got {queries.shape}")
        
        # Initialize packed arrays
        total_slots = self.layout.ckks_params.simd_slots
        packed_q = np.zeros(total_slots, dtype=complex)
        packed_k = np.zeros(total_slots, dtype=complex)
        packed_v = np.zeros(total_slots, dtype=complex)
        
        # Pack each head's data
        for head in range(self.layout.h):
            head_start, head_end = self.layout.get_head_slot_range(head)
            
            # Flatten and pack the head's data
            # For simplicity, we'll pack in sequence order
            # In practice, this would follow the tile structure
            q_flat = queries[head].flatten()
            k_flat = keys[head].flatten()
            v_flat = values[head].flatten()
            
            # Take only as many elements as we have slots for this head
            max_elements = min(len(q_flat), self.layout.elements_per_head)
            
            packed_q[head_start:head_start + max_elements] = q_flat[:max_elements]
            packed_k[head_start:head_start + max_elements] = k_flat[:max_elements]
            packed_v[head_start:head_start + max_elements] = v_flat[:max_elements]
        
        return {
            'queries': packed_q,
            'keys': packed_k,
            'values': packed_v
        }
    
    def simulate_parallel_attention(self, packed_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Simulate parallel multi-head attention computation
        
        Returns performance metrics
        """
        start_time = time.time()
        
        # Simulate Q*K^T computation for all heads in parallel
        qk_multiplications = 0
        
        for head in range(self.layout.h):
            head_start, head_end = self.layout.get_head_slot_range(head)
            
            # Simulate Winograd tile processing
            for tile in range(self.layout.T):
                tile_start, tile_end = self.layout.get_tile_slot_range(head, tile)
                
                # Count Winograd multiplications per tile
                qk_multiplications += self.layout.wg_config.multiplications_per_tile
        
        qk_time = time.time() - start_time
        
        # Simulate P*V computation (similar structure)
        pv_start = time.time()
        pv_multiplications = qk_multiplications  # Same structure
        pv_time = time.time() - pv_start
        
        total_time = qk_time + pv_time
        total_multiplications = qk_multiplications + pv_multiplications
        
        # Calculate theoretical baseline multiplications
        baseline_multiplications = 2 * self.layout.h * self.layout.T * \
                                 self.layout.wg_config.m * self.layout.wg_config.r
        
        return {
            'total_time': total_time,
            'total_multiplications': total_multiplications,
            'baseline_multiplications': baseline_multiplications,
            'multiplication_reduction': 1 - (total_multiplications / baseline_multiplications),
            'heads_processed': self.layout.h,
            'tiles_per_head': self.layout.T,
            'parallelization_efficiency': self.layout.h / max(1, total_time * 1000)  # heads per ms
        }
    
    def analyze_rotation_requirements(self) -> Dict[str, int]:
        """Analyze rotation key requirements"""
        
        # Count unique rotations needed
        unique_rotations = set()
        
        # Diagonal gathering rotations
        for r in range(self.layout.wg_config.r):
            unique_rotations.add(r)
        
        # Tile boundary rotations
        unique_rotations.update(self.rotation_patterns['tile_boundary'])
        
        # Head boundary rotations (for boundary crossing)
        unique_rotations.update(self.rotation_patterns['head_boundary'])
        
        # Baby-step/Giant-step decomposition
        bsgs_base = int(np.sqrt(self.layout.ckks_params.simd_slots))
        baby_steps = list(range(bsgs_base))
        giant_steps = [i * bsgs_base for i in range(self.layout.ckks_params.simd_slots // bsgs_base)]
        
        hierarchical_keys = set(baby_steps + giant_steps)
        
        return {
            'direct_rotations': len(unique_rotations),
            'hierarchical_keys': len(hierarchical_keys),
            'total_with_hierarchy': len(hierarchical_keys | unique_rotations),
            'bsgs_base': bsgs_base,
            'reduction_ratio': len(hierarchical_keys) / self.layout.ckks_params.simd_slots
        }

def test_phase4_implementation():
    """Test Phase 4 implementation with various configurations"""
    
    print("=== Phase 4: SIMD Slot Layout and Multi-Head Processing Test ===\n")
    
    # Test configurations
    configs = [
        {
            'name': 'Small Model F(2,3)',
            'h': 8, 'L': 64, 'd_model': 512,
            'winograd': WinogradConfig(m=2, r=3),
            'ckks': CKKSParams(N=2**14)
        },
        {
            'name': 'Medium Model F(4,3)', 
            'h': 12, 'L': 128, 'd_model': 768,
            'winograd': WinogradConfig(m=4, r=3),
            'ckks': CKKSParams(N=2**15)
        },
        {
            'name': 'Large Model F(2,3)',
            'h': 16, 'L': 256, 'd_model': 1024,
            'winograd': WinogradConfig(m=2, r=3),
            'ckks': CKKSParams(N=2**16)
        }
    ]
    
    results = []
    
    for config in configs:
        print(f"\n--- Testing {config['name']} ---")
        
        try:
            # Create layout
            layout = SIMDSlotLayout(
                h=config['h'],
                L=config['L'], 
                d_model=config['d_model'],
                winograd_config=config['winograd'],
                ckks_params=config['ckks']
            )
            
            print(f"✓ Layout created successfully")
            print(f"  - Heads: {layout.h}")
            print(f"  - Tiles per head: {layout.T}")
            print(f"  - Elements per head: {layout.elements_per_head}")
            print(f"  - Total slots used: {layout.total_elements}")
            print(f"  - Packing efficiency: {layout.calculate_packing_efficiency():.2%}")
            
            # Test slot indexing
            test_head, test_tile, test_element = 0, 0, 0
            slot_idx = layout.slot_index(test_head, test_tile, test_element)
            reverse_result = layout.reverse_slot_index(slot_idx)
            
            assert reverse_result == (test_head, test_tile, test_element), \
                "Slot indexing round-trip failed"
            print(f"✓ Slot indexing test passed")
            
            # Create processor
            processor = MultiHeadWinogradProcessor(layout)
            
            # Create dummy data
            queries = np.random.randn(layout.h, layout.L, layout.d_k) + \
                     1j * np.random.randn(layout.h, layout.L, layout.d_k)
            keys = np.random.randn(layout.h, layout.L, layout.d_k) + \
                  1j * np.random.randn(layout.h, layout.L, layout.d_k)  
            values = np.random.randn(layout.h, layout.L, layout.d_k) + \
                    1j * np.random.randn(layout.h, layout.L, layout.d_k)
            
            # Test packing
            packed_data = processor.pack_multi_head_data(queries, keys, values)
            print(f"✓ Multi-head data packing successful")
            
            # Test parallel attention simulation
            metrics = processor.simulate_parallel_attention(packed_data)
            print(f"✓ Parallel attention simulation completed")
            print(f"  - Multiplication reduction: {metrics['multiplication_reduction']:.1%}")
            print(f"  - Winograd multiplications: {metrics['total_multiplications']}")
            print(f"  - Baseline multiplications: {metrics['baseline_multiplications']}")
            
            # Analyze rotations
            rotation_analysis = processor.analyze_rotation_requirements()
            print(f"✓ Rotation analysis completed")
            print(f"  - Direct rotations needed: {rotation_analysis['direct_rotations']}")
            print(f"  - Hierarchical keys: {rotation_analysis['hierarchical_keys']}")
            print(f"  - Key reduction ratio: {rotation_analysis['reduction_ratio']:.3f}")
            
            results.append({
                'config': config['name'],
                'packing_efficiency': layout.calculate_packing_efficiency(),
                'multiplication_reduction': metrics['multiplication_reduction'],
                'rotation_keys': rotation_analysis['hierarchical_keys'],
                'success': True
            })
            
            # Visualize the first configuration
            if config == configs[0]:
                print(f"\nVisualizing layout for {config['name']}:")
                layout.visualize_layout(max_display_slots=256)
            
        except Exception as e:
            print(f"✗ Error with {config['name']}: {str(e)}")
            results.append({
                'config': config['name'],
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n=== Summary ===")
    for result in results:
        if result['success']:
            print(f"{result['config']}:")
            print(f"  - Packing efficiency: {result['packing_efficiency']:.2%}")
            print(f"  - Multiplication reduction: {result['multiplication_reduction']:.1%}")
            print(f"  - Rotation keys: {result['rotation_keys']}")
        else:
            print(f"{result['config']}: FAILED - {result['error']}")
    
    return results

# Additional utility functions for integration with existing phases

def compare_with_baseline(layout: SIMDSlotLayout) -> Dict[str, float]:
    """Compare Winograd approach with baseline diagonal method"""
    
    # Baseline: standard diagonal method
    baseline_mults_per_tile = layout.wg_config.m * layout.wg_config.r
    baseline_total_mults = layout.h * layout.T * baseline_mults_per_tile
    
    # Winograd approach  
    winograd_mults_per_tile = layout.wg_config.multiplications_per_tile
    winograd_total_mults = layout.h * layout.T * winograd_mults_per_tile
    
    # Rotation costs (simplified model)
    baseline_rotations = layout.h * layout.T * layout.wg_config.r
    winograd_rotations = layout.h * max(1, layout.T // 10)  # Boundary rotations only
    
    return {
        'multiplication_reduction': 1 - (winograd_total_mults / baseline_total_mults),
        'rotation_reduction': 1 - (winograd_rotations / baseline_rotations),
        'winograd_mults': winograd_total_mults,
        'baseline_mults': baseline_total_mults,
        'winograd_rotations': winograd_rotations,
        'baseline_rotations': baseline_rotations
    }

def estimate_memory_footprint(layout: SIMDSlotLayout) -> Dict[str, float]:
    """Estimate memory requirements in MB"""
    
    # Weight cache: h * T * (m + r - 1) * log2(Q) * N bits
    weight_cache_bits = (layout.h * layout.T * layout.wg_config.multiplications_per_tile * 
                        layout.ckks_params.log_Q * layout.ckks_params.N)
    
    # Rotation keys: approximately sqrt(N) keys * log(Q) * N bits each
    bsgs_keys = int(np.sqrt(layout.ckks_params.simd_slots))
    rotation_keys_bits = bsgs_keys * layout.ckks_params.log_Q * layout.ckks_params.N
    
    # HRF pre-rotation cache: r rotations * N/2 slots * log(Q) bits
    hrf_cache_bits = layout.wg_config.r * layout.ckks_params.simd_slots * layout.ckks_params.log_Q
    
    return {
        'weight_cache_mb': weight_cache_bits / (8 * 1024 * 1024),
        'rotation_keys_mb': rotation_keys_bits / (8 * 1024 * 1024), 
        'hrf_cache_mb': hrf_cache_bits / (8 * 1024 * 1024),
        'total_mb': (weight_cache_bits + rotation_keys_bits + hrf_cache_bits) / (8 * 1024 * 1024)
    }

if __name__ == "__main__":
    # Run the comprehensive test
    test_results = test_phase4_implementation()