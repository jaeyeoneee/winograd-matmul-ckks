import numpy as np
from typing import List, Tuple, Iterator, Optional
import math
from dataclasses import dataclass

@dataclass
class TileInfo:
    """Information about a single tile"""
    tile_id: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    tile_shape: Tuple[int, int]
    is_boundary: bool
    padding_needed: Tuple[int, int]  # (row_padding, col_padding)

class TileDecomposer:
    """
    Tile decomposition algorithm for Winograd-compatible matrix multiplication
    Based on paper Section 3.2: Diagonal Method Extension to Tile Blocks
    """
    
    def __init__(self, tile_m: int, tile_r: int):
        """
        Initialize tile decomposer
        
        Args:
            tile_m: Tile height (output size in Winograd F(m,r))
            tile_r: Tile width (filter size in Winograd F(m,r))
        """
        self.tile_m = tile_m
        self.tile_r = tile_r
        self.config_name = f"F({tile_m},{tile_r})"
        
        print(f"Initialized Tile Decomposer for {self.config_name}")
        print(f"Tile dimensions: {tile_m} × {tile_r}")
    
    def decompose_matrix(self, matrix_shape: Tuple[int, int], 
                        matrix_type: str = "input") -> List[TileInfo]:
        """
        Decompose matrix into Winograd-compatible tiles
        
        Args:
            matrix_shape: (rows, cols) of the matrix
            matrix_type: "input" or "weight" for different tiling strategies
        Returns:
            List of TileInfo objects describing each tile
        """
        rows, cols = matrix_shape
        
        if matrix_type == "input":
            # For input matrices, tile by output dimensions (m × d)
            tile_height, tile_width = self.tile_m, cols
            step_height, step_width = self.tile_m, cols
        elif matrix_type == "weight":
            # For weight matrices, tile by filter dimensions (d × r)
            tile_height, tile_width = rows, self.tile_r
            step_height, step_width = rows, self.tile_r
        else:
            # General tiling
            tile_height, tile_width = self.tile_m, self.tile_r
            step_height, step_width = self.tile_m, self.tile_r
        
        tiles = []
        tile_id = 0
        
        # Calculate number of tiles needed
        num_tiles_row = math.ceil(rows / step_height)
        num_tiles_col = math.ceil(cols / step_width)
        
        print(f"\nDecomposing {matrix_type} matrix {matrix_shape}:")
        print(f"Tile size: {tile_height} × {tile_width}")
        print(f"Number of tiles: {num_tiles_row} × {num_tiles_col} = {num_tiles_row * num_tiles_col}")
        
        for i in range(num_tiles_row):
            for j in range(num_tiles_col):
                # Calculate tile boundaries
                row_start = i * step_height
                row_end = min((i + 1) * step_height, rows)
                col_start = j * step_width
                col_end = min((j + 1) * step_width, cols)
                
                # Actual tile shape
                actual_height = row_end - row_start
                actual_width = col_end - col_start
                tile_shape = (actual_height, actual_width)
                
                # Check if this is a boundary tile requiring padding
                is_boundary = (actual_height < tile_height) or (actual_width < tile_width)
                
                # Calculate padding needed
                row_padding = max(0, tile_height - actual_height)
                col_padding = max(0, tile_width - actual_width)
                padding_needed = (row_padding, col_padding)
                
                tile_info = TileInfo(
                    tile_id=tile_id,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    tile_shape=tile_shape,
                    is_boundary=is_boundary,
                    padding_needed=padding_needed
                )
                
                tiles.append(tile_info)
                tile_id += 1
        
        return tiles
    
    def extract_tile(self, matrix: np.ndarray, tile_info: TileInfo, 
                    pad_to_tile_size: bool = True) -> np.ndarray:
        """
        Extract a tile from matrix with optional padding
        
        Args:
            matrix: Input matrix
            tile_info: TileInfo describing the tile to extract
            pad_to_tile_size: Whether to pad tile to standard size
        Returns:
            Extracted (and optionally padded) tile
        """
        # Extract the tile region
        tile = matrix[tile_info.row_start:tile_info.row_end,
                     tile_info.col_start:tile_info.col_end]
        
        if pad_to_tile_size and tile_info.is_boundary:
            # Pad tile to standard size
            row_pad, col_pad = tile_info.padding_needed
            
            if row_pad > 0 or col_pad > 0:
                padded_tile = np.zeros((self.tile_m, self.tile_r))
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                return padded_tile
        
        return tile
    
    def reconstruct_matrix(self, tiles: List[np.ndarray], tile_infos: List[TileInfo],
                          original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstruct matrix from tiles
        
        Args:
            tiles: List of tile arrays
            tile_infos: List of TileInfo objects
            original_shape: Original matrix shape
        Returns:
            Reconstructed matrix
        """
        reconstructed = np.zeros(original_shape)
        
        for tile, tile_info in zip(tiles, tile_infos):
            # Extract the valid portion of the tile (remove padding)
            valid_height = tile_info.row_end - tile_info.row_start
            valid_width = tile_info.col_end - tile_info.col_start
            
            valid_tile = tile[:valid_height, :valid_width]
            
            # Place back in the reconstructed matrix
            reconstructed[tile_info.row_start:tile_info.row_end,
                        tile_info.col_start:tile_info.col_end] = valid_tile
        
        return reconstructed
    
    def get_multiplication_tiles(self, left_shape: Tuple[int, int], 
                               right_shape: Tuple[int, int]) -> Tuple[List[TileInfo], List[TileInfo]]:
        """
        Get tile decomposition for matrix multiplication: left @ right
        
        Args:
            left_shape: Shape of left matrix (L, d)
            right_shape: Shape of right matrix (d, R)
        Returns:
            Tuple of (left_tiles, right_tiles)
        """
        L, d = left_shape
        d_check, R = right_shape
        
        if d != d_check:
            raise ValueError(f"Matrix dimension mismatch: {left_shape} @ {right_shape}")
        
        print(f"\nPlanning matrix multiplication: {left_shape} @ {right_shape}")
        
        # Left matrix tiles: partition by rows (output dimension)
        left_tiles = []
        num_left_tiles = math.ceil(L / self.tile_m)
        
        for i in range(num_left_tiles):
            row_start = i * self.tile_m
            row_end = min((i + 1) * self.tile_m, L)
            
            tile_info = TileInfo(
                tile_id=i,
                row_start=row_start,
                row_end=row_end,
                col_start=0,
                col_end=d,
                tile_shape=(row_end - row_start, d),
                is_boundary=(row_end - row_start) < self.tile_m,
                padding_needed=(max(0, self.tile_m - (row_end - row_start)), 0)
            )
            left_tiles.append(tile_info)
        
        # Right matrix tiles: partition by columns (output dimension)
        right_tiles = []
        num_right_tiles = math.ceil(R / self.tile_r)
        
        for j in range(num_right_tiles):
            col_start = j * self.tile_r
            col_end = min((j + 1) * self.tile_r, R)
            
            tile_info = TileInfo(
                tile_id=j,
                row_start=0,
                row_end=d,
                col_start=col_start,
                col_end=col_end,
                tile_shape=(d, col_end - col_start),
                is_boundary=(col_end - col_start) < self.tile_r,
                padding_needed=(0, max(0, self.tile_r - (col_end - col_start)))
            )
            right_tiles.append(tile_info)
        
        print(f"Left matrix tiles: {len(left_tiles)}")
        print(f"Right matrix tiles: {len(right_tiles)}")
        print(f"Total tile pairs: {len(left_tiles)} × {len(right_tiles)} = {len(left_tiles) * len(right_tiles)}")
        
        return left_tiles, right_tiles
    
    def estimate_operations(self, left_shape: Tuple[int, int], 
                          right_shape: Tuple[int, int]) -> dict:
        """
        Estimate operation counts for tiled vs direct multiplication
        
        Args:
            left_shape: Left matrix shape
            right_shape: Right matrix shape
        Returns:
            Dictionary with operation counts
        """
        L, d = left_shape
        d_check, R = right_shape
        
        if d != d_check:
            raise ValueError(f"Matrix dimension mismatch: {left_shape} @ {right_shape}")
        
        # Direct multiplication operations
        direct_ops = L * d * R
        
        # Tiled operations
        left_tiles, right_tiles = self.get_multiplication_tiles(left_shape, right_shape)
        
        total_tile_pairs = len(left_tiles) * len(right_tiles)
        
        # Each tile pair performs tile_m × d × tile_r operations (direct)
        # With Winograd: tile_m × d × tile_r -> d × (tile_m + tile_r - 1)
        direct_ops_per_tile = self.tile_m * d * self.tile_r
        winograd_ops_per_tile = d * (self.tile_m + self.tile_r - 1)
        
        total_direct_tiled = total_tile_pairs * direct_ops_per_tile
        total_winograd_tiled = total_tile_pairs * winograd_ops_per_tile
        
        # Calculate reductions
        tiling_reduction = 1 - (total_direct_tiled / direct_ops) if direct_ops > 0 else 0
        winograd_reduction = 1 - (winograd_ops_per_tile / direct_ops_per_tile) if direct_ops_per_tile > 0 else 0
        total_reduction = 1 - (total_winograd_tiled / direct_ops) if direct_ops > 0 else 0
        
        return {
            'direct_operations': direct_ops,
            'direct_tiled_operations': total_direct_tiled,
            'winograd_tiled_operations': total_winograd_tiled,
            'tile_pairs': total_tile_pairs,
            'ops_per_tile_direct': direct_ops_per_tile,
            'ops_per_tile_winograd': winograd_ops_per_tile,
            'tiling_overhead': tiling_reduction,
            'winograd_reduction_per_tile': winograd_reduction,
            'total_reduction': total_reduction
        }

def test_tile_decomposition():
    """Test tile decomposition functionality"""
    print("=== Testing Tile Decomposition Algorithm ===\n")
    
    # Test with F(2,3) configuration
    decomposer = TileDecomposer(tile_m=2, tile_r=3)
    
    print("\n1. Basic tile decomposition test:")
    
    # Test matrix shapes
    test_matrices = [
        ((4, 6), "Small matrix"),
        ((7, 10), "Medium matrix with boundary tiles"),
        ((16, 32), "Larger matrix"),
        ((5, 3), "Exact tile fit"),
    ]
    
    for matrix_shape, description in test_matrices:
        print(f"\n--- {description}: {matrix_shape} ---")
        tiles = decomposer.decompose_matrix(matrix_shape, "general")
        
        # Count boundary tiles
        boundary_tiles = sum(1 for tile in tiles if tile.is_boundary)
        print(f"Boundary tiles: {boundary_tiles}/{len(tiles)}")
        
        # Test extraction and reconstruction
        test_matrix = np.random.randn(*matrix_shape)
        
        # Extract all tiles
        extracted_tiles = []
        for tile_info in tiles:
            tile = decomposer.extract_tile(test_matrix, tile_info, pad_to_tile_size=True)
            extracted_tiles.append(tile)
            
        print(f"Extracted {len(extracted_tiles)} tiles")
        
        # Test reconstruction
        reconstructed = decomposer.reconstruct_matrix(extracted_tiles, tiles, matrix_shape)
        reconstruction_error = np.max(np.abs(test_matrix - reconstructed))
        print(f"Reconstruction error: {reconstruction_error:.2e}")
    
    print("\n2. Matrix multiplication tile planning:")
    
    # Test multiplication scenarios
    mult_tests = [
        ((8, 16), (16, 12), "Standard multiplication"),
        ((10, 20), (20, 6), "Tall × wide"),
        ((4, 8), (8, 6), "Small matrices"),
        ((100, 50), (50, 75), "Large matrices"),
    ]
    
    for left_shape, right_shape, description in mult_tests:
        print(f"\n--- {description}: {left_shape} @ {right_shape} ---")
        
        try:
            ops_analysis = decomposer.estimate_operations(left_shape, right_shape)
            
            print(f"Direct operations: {ops_analysis['direct_operations']:,}")
            print(f"Winograd tiled operations: {ops_analysis['winograd_tiled_operations']:,}")
            print(f"Tile pairs: {ops_analysis['tile_pairs']}")
            print(f"Total reduction: {ops_analysis['total_reduction']:.1%}")
            
        except Exception as e:
            print(f"Error in analysis: {e}")
    
    print("\n3. Tile size optimization analysis:")
    
    # Compare different tile configurations
    test_matrix_shape = (32, 64)
    test_right_shape = (64, 48)
    
    tile_configs = [(2, 3), (4, 3), (2, 2), (4, 4)]
    
    print(f"\nOptimizing for {test_matrix_shape} @ {test_right_shape}:")
    
    results = []
    for m, r in tile_configs:
        try:
            decomposer_test = TileDecomposer(tile_m=m, tile_r=r)
            ops = decomposer_test.estimate_operations(test_matrix_shape, test_right_shape)
            
            results.append({
                'config': f"F({m},{r})",
                'tile_pairs': ops['tile_pairs'],
                'total_reduction': ops['total_reduction'],
                'winograd_reduction': ops['winograd_reduction_per_tile']
            })
            
        except Exception as e:
            print(f"Error with F({m},{r}): {e}")
    
    # Sort by total reduction
    results.sort(key=lambda x: x['total_reduction'], reverse=True)
    
    print(f"\nTile configuration comparison:")
    print(f"{'Config':<8} {'Tiles':<8} {'Winograd':<10} {'Total':<10}")
    print("-" * 40)
    for result in results:
        print(f"{result['config']:<8} {result['tile_pairs']:<8} "
              f"{result['winograd_reduction']:<9.1%} {result['total_reduction']:<9.1%}")
    
    return decomposer

def visualize_tiling(decomposer: TileDecomposer, matrix_shape: Tuple[int, int]):
    """Visualize matrix tiling pattern"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    tiles = decomposer.decompose_matrix(matrix_shape, "general")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Draw matrix outline
    rect = patches.Rectangle((0, 0), matrix_shape[1], matrix_shape[0], 
                           linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
    # Draw tiles
    colors = plt.cm.Set3(np.linspace(0, 1, len(tiles)))
    
    for i, tile in enumerate(tiles):
        # Note: matplotlib uses (x, y) coordinates, so we swap col/row
        rect = patches.Rectangle(
            (tile.col_start, matrix_shape[0] - tile.row_end),  # Bottom-left corner
            tile.col_end - tile.col_start,  # Width
            tile.row_end - tile.row_start,  # Height
            linewidth=1,
            edgecolor='red' if tile.is_boundary else 'blue',
            facecolor=colors[i],
            alpha=0.3
        )
        ax.add_patch(rect)
        
        # Add tile ID
        center_x = (tile.col_start + tile.col_end) / 2
        center_y = matrix_shape[0] - (tile.row_start + tile.row_end) / 2
        ax.text(center_x, center_y, str(tile.tile_id), 
               ha='center', va='center', fontweight='bold')
    
    ax.set_xlim(-0.5, matrix_shape[1] + 0.5)
    ax.set_ylim(-0.5, matrix_shape[0] + 0.5)
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    ax.set_title(f'Matrix Tiling Pattern ({decomposer.config_name})\n'
                f'Matrix: {matrix_shape}, Tile: {decomposer.tile_m}×{decomposer.tile_r}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add legend
    boundary_patch = patches.Patch(color='red', alpha=0.3, label='Boundary tiles')
    regular_patch = patches.Patch(color='blue', alpha=0.3, label='Regular tiles')
    ax.legend(handles=[regular_patch, boundary_patch])
    
    plt.tight_layout()
    plt.show()

# Run tests if executed directly
if __name__ == "__main__":
    decomposer = test_tile_decomposition()
    
    # Visualize an example
    print("\n4. Visualization example:")
    try:
        visualize_tiling(decomposer, (8, 12))
    except ImportError:
        print("Matplotlib not available for visualization")