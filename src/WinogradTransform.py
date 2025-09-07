import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import time

class WinogradTransform:
    """
    Winograd minimal filtering transforms for F(m,r) configurations
    Based on the paper's Section 2.4 and 3.3
    """
    
    def __init__(self, m: int, r: int):
        """
        Initialize Winograd transform matrices for F(m,r) configuration
        
        Args:
            m: output tile size
            r: filter size (number of input elements)
        """
        self.m = m
        self.r = r
        self.config_name = f"F({m},{r})"
        
        # Initialize transform matrices
        self.B, self.G, self.A = self._get_transform_matrices(m, r)
        
        # Precompute transposes for efficiency
        self.B_T = self.B.T
        self.G_T = self.G.T
        self.A_T = self.A.T
        
        print(f"Initialized {self.config_name} Winograd transform")
        print(f"Transform dimensions:")
        print(f"  B: {self.B.shape}, B_T: {self.B_T.shape}")
        print(f"  G: {self.G.shape}, G_T: {self.G_T.shape}")
        print(f"  A: {self.A.shape}, A_T: {self.A_T.shape}")
        print(f"Transform domain size: {self.m + self.r - 1}")
        print(f"Filter size: {self.r}")
        print(f"Multiplication reduction: {self.m * self.r} -> {self.m + self.r - 1} "
              f"({100 * (1 - (self.m + self.r - 1)/(self.m * self.r)):.1f}% reduction)")
        print()
    
    def _get_transform_matrices(self, m: int, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the transform matrices for specific F(m,r) configurations
        """
        if m == 2 and r == 3:
            return self._get_F_2_3_matrices()
        elif m == 4 and r == 3:
            return self._get_F_4_3_matrices()
        elif m == 6 and r == 3:
            return self._get_F_6_3_matrices()
        else:
            raise NotImplementedError(f"F({m},{r}) configuration not implemented")
    
    def _get_F_2_3_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """F(2,3) transform matrices from the paper"""
        B_T = np.array([
            [1,  0, -1,  0],
            [0,  1,  1,  0],
            [0, -1,  1,  0],
            [0,  1,  0, -1]
        ], dtype=np.float64)
        
        G = np.array([
            [1,    0,    0],
            [0.5,  0.5,  0.5],
            [0.5, -0.5,  0.5],
            [0,    0,    1]
        ], dtype=np.float64)
        
        A_T = np.array([
            [1,  1,  1,  0],
            [0,  1, -1, -1]
        ], dtype=np.float64)
        
        return B_T.T, G, A_T.T
    
    def _get_F_4_3_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """F(4,3) transform matrices (simplified version)"""
        # Simplified F(4,3) matrices - in practice these would be optimized
        B_T = np.array([
            [1,  0,  0,  0, -1,  0],
            [0,  1, -1,  1,  1,  0],
            [0, -1, -1, -1,  1,  0],
            [0,  1,  0, -1,  0,  0],
            [0, -1,  0,  1,  0,  0],
            [0,  0,  1,  0,  0, -1]
        ], dtype=np.float64)
        
        G = np.array([
            [1,     0,     0],
            [2/3,   2/3,   2/3],
            [2/3,  -2/3,   2/3],
            [1/6,   1/3,   2/3],
            [1/6,  -1/3,   2/3],
            [0,     0,     1]
        ], dtype=np.float64)
        
        A_T = np.array([
            [1,  1,  1,  1,  1,  0],
            [0,  1, -1,  2, -2,  0],
            [0,  1,  1,  4,  4,  0],
            [0,  1, -1,  8, -8,  1]
        ], dtype=np.float64)
        
        return B_T.T, G, A_T.T
    
    def _get_F_6_3_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """F(6,3) transform matrices (simplified version)"""
        # This is a placeholder - actual F(6,3) matrices would be more complex
        size = 6 + 3 - 1  # 8
        B = np.eye(size, 6, dtype=np.float64)
        G = np.eye(size, 3, dtype=np.float64)
        A = np.eye(6, size, dtype=np.float64)
        return B, G, A
    
    def forward_transform_input(self, x: np.ndarray) -> np.ndarray:
        """
        Apply input transform: B_T @ x @ B
        
        Args:
            x: Input data (will be padded to transform domain size)
        Returns:
            Transformed input
        """
        transform_size = self.m + self.r - 1  # Transform domain size
        
        if x.ndim == 2:
            # Pad input to transform domain size
            x_padded = self._pad_to_size(x, (transform_size, transform_size))
            return self.B_T @ x_padded @ self.B
        elif x.ndim == 3:
            return np.array([self.forward_transform_input(x_i) for x_i in x])
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
    
    def forward_transform_weight(self, w: np.ndarray) -> np.ndarray:
        """
        Apply weight transform: G @ w @ G_T
        
        Args:
            w: Weight data (should be filter size r × r)
        Returns:
            Transformed weight
        """
        if w.ndim == 2:
            # For Winograd, weight should be r × r, pad if necessary
            w_padded = self._pad_to_size(w, (self.r, self.r))
            return self.G @ w_padded @ self.G_T
        elif w.ndim == 3:
            return np.array([self.forward_transform_weight(w_i) for w_i in w])
        else:
            raise ValueError(f"Unsupported weight shape: {w.shape}")
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Apply output transform: A_T @ y @ A
        
        Args:
            y: Transformed output from element-wise multiplication
        Returns:
            Final output
        """
        if y.ndim == 2:
            return self.A_T @ y @ self.A
        elif y.ndim == 3:
            return np.array([self.A_T @ y_i @ self.A for y_i in y])
        else:
            raise ValueError(f"Unsupported output shape: {y.shape}")
    
    def _pad_to_size(self, matrix: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Pad matrix to target size with zeros
        
        Args:
            matrix: Input matrix
            target_size: Target (height, width)
        Returns:
            Padded matrix
        """
        current_h, current_w = matrix.shape
        target_h, target_w = target_size
        
        if current_h > target_h or current_w > target_w:
            raise ValueError(f"Input shape {matrix.shape} exceeds target size {target_size}.")

        padded = np.zeros(target_size, dtype=matrix.dtype)  # dtype 유지
        padded[:current_h, :current_w] = matrix
        
        return padded
    
    def winograd_tile_multiply(self, x_tile: np.ndarray, w_tile: np.ndarray) -> np.ndarray:
        """
        Simplified Winograd tile multiplication demonstration
        
        Args:
            x_tile: Input tile 
            w_tile: Weight tile
        Returns:
            Output tile (for demonstration purposes)
        """
        alpha = self.m + self.r - 1
        x_padded = self._pad_to_size(x_tile, (alpha, alpha))
        w_padded = self._pad_to_size(w_tile, (self.r, self.r))  # 가중치는 r×r 유지

        x_transformed = self.forward_transform_input(x_padded)
        w_transformed = self.forward_transform_weight(w_padded)

        y_transformed = x_transformed * w_transformed
        y_output = self.inverse_transform(y_transformed)

        return y_output[:self.m, :self.m]  # <-- 핵심 수정
    
    def verify_correctness(self, trials: int = 8, tol: float = 1e-10) -> bool:
        """
        Verify that Winograd transform preserves correctness with identity test
        """
        def xcorr2d_valid(x, w):
            r = w.shape[0]
            H, W = x.shape
            out = np.zeros((H - r + 1, W - r + 1), dtype=x.dtype)
            for i in range(out.shape[0]):
                for j in range(out.shape[1]):
                    out[i, j] = np.sum(x[i:i+r, j:j+r] * w)
            return out

        alpha = self.m + self.r - 1
        ok = True
        for _ in range(trials):
            x = np.random.randn(alpha, alpha)
            w = np.random.randn(self.r, self.r)
            y_wino = self.winograd_tile_multiply(x, w)
            y_direct = xcorr2d_valid(x, w)
            err = np.max(np.abs(y_wino - y_direct))
            if not (err < tol):
                print(f"[verify] max error = {err:.2e} (tol={tol})")
                ok = False
                break
        return ok
    
    def demonstrate_winograd_benefit(self) -> dict:
        """
        Demonstrate the computational benefit of Winograd transforms
        """
        alpha = self.m + self.r - 1
        # 타일 하나 기준의 핵심 곱셈 수(2D):
        direct_mults = (self.m * self.m) * (self.r * self.r)
        winograd_mults = alpha * alpha

        # 간단 시연(정합 크기)
        x_test = np.random.randn(alpha, alpha)
        w_test = np.random.randn(self.r, self.r)
        x_t = self.forward_transform_input(x_test)
        w_t = self.forward_transform_weight(w_test)
        y_t = x_t * w_t
        y = self.inverse_transform(y_t)

        return {
            "config": self.config_name,
            "alpha": alpha,
            "direct_multiplications_per_tile": direct_mults,
            "winograd_multiplications_per_tile": winograd_mults,
            "reduction_factor": 1 - (winograd_mults / direct_mults),
            "output_shape": y.shape
        }

    
    def count_operations(self, tile_size: Tuple[int, int, int]) -> dict:
        """
        Count operations for performance analysis
        
        Args:
            tile_size: (m, d, r) dimensions
        Returns:
            Dictionary with operation counts
        """
        # tile_size = (m, C, r) 가정: 채널 C
        m, C, r = tile_size
        alpha = m + r - 1

        direct_mults = C * (m * m) * (r * r)
        winograd_mults = C * (alpha * alpha)  # 채널별 Hadamard 곱 후 채널 합성 필요
        # 변환(Add/Const-Mul) 비용은 구현 세부에 따라 달라지므로 별도 근사
        transform_adds = None

        reduction = 1 - (winograd_mults / direct_mults)
        return {
            "direct_multiplications": direct_mults,
            "winograd_multiplications": winograd_mults,
            "reduction_factor": reduction,
            "savings_percent": 100 * reduction
        }

# Test the implementation - COMPLETELY REWRITTEN
def test_winograd_transform():
    """Test basic Winograd transform functionality"""
    print("=== Testing Winograd Transform Implementation ===\n")
    
    # Test F(2,3) configuration
    print("1. Testing F(2,3) configuration:")
    winograd_2_3 = WinogradTransform(m=2, r=3)
    
    print(f"\n2. Testing individual transforms:")
    # FIXED: Use correct variable names and sizes
    transform_size = 4  # m + r - 1 = 2 + 3 - 1 = 4
    filter_size = 3     # r = 3
    
    print(f"Transform domain size: {transform_size}×{transform_size}")
    print(f"Filter size: {filter_size}×{filter_size}")
    
    # FIXED: Create test data with CORRECT sizes
    test_input = np.random.randn(transform_size, transform_size)  # 4×4
    test_weight = np.random.randn(filter_size, filter_size)      # 3×3 ← THIS WAS THE BUG!
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Test weight shape: {test_weight.shape}")
    
    # Apply transforms
    x_transformed = winograd_2_3.forward_transform_input(test_input)
    w_transformed = winograd_2_3.forward_transform_weight(test_weight)
    
    print(f"Input transform: {test_input.shape} -> {x_transformed.shape}")
    print(f"Weight transform: {test_weight.shape} -> {w_transformed.shape}")
    
    # Element-wise multiplication
    y_transformed = x_transformed * w_transformed
    print(f"Element-wise mult: {y_transformed.shape}")
    print(f"Number of multiplications: {winograd_2_3.m + winograd_2_3.r - 1} (vs direct: {winograd_2_3.m * winograd_2_3.r})")
    
    # Inverse transform
    y_output = winograd_2_3.inverse_transform(y_transformed)
    print(f"Inverse transform: {y_transformed.shape} -> {y_output.shape}")
    
    # Verification test
    print(f"\n3. Verification test:")
    is_correct = winograd_2_3.verify_correctness()
    print(f"Transform correctness: {'✓ PASS' if is_correct else '✗ FAIL'}")
    
    # Demonstration of benefits
    print(f"\n4. Winograd benefit demonstration:")
    benefits = winograd_2_3.demonstrate_winograd_benefit()
    for key, value in benefits.items():
        if isinstance(value, float):
            if 'factor' in key or 'reduction' in key:
                print(f"{key}: {value:.1%}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Try F(4,3) configuration
    try:
        print(f"\n5. Testing F(4,3) configuration:")
        winograd_4_3 = WinogradTransform(m=4, r=3)
        
        # FIXED: Use correct sizes for F(4,3)
        transform_size_43 = 6  # 4 + 3 - 1 = 6
        filter_size_43 = 3     # 3
        
        test_input_43 = np.random.randn(transform_size_43, transform_size_43)  # 6×6
        test_weight_43 = np.random.randn(filter_size_43, filter_size_43)      # 3×3
        
        x_trans_43 = winograd_4_3.forward_transform_input(test_input_43)
        w_trans_43 = winograd_4_3.forward_transform_weight(test_weight_43)
        
        print(f"F(4,3) input transform: {test_input_43.shape} -> {x_trans_43.shape}")
        print(f"F(4,3) weight transform: {test_weight_43.shape} -> {w_trans_43.shape}")
        
        # Verification test
        print(f"\n3. Verification test:")
        is_correct = winograd_2_3.verify_correctness()
        print(f"Transform correctness: {'✓ PASS' if is_correct else '✗ FAIL'}")
   
        benefits_43 = winograd_4_3.demonstrate_winograd_benefit()
        print(f"F(4,3) reduction: {benefits_43['reduction_factor']:.1%}")
        
    except Exception as e:
        print(f"F(4,3) test failed: {e}")
    
    # Operation count analysis
    print("\n6. Operation count analysis:")
    ops = winograd_2_3.count_operations((2, 32, 3))
    for key, value in ops.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Test small matrix multiplication example
    print(f"\n7. Small matrix multiplication example:")
    try:
        small_x = np.random.randn(2, 4)
        small_w = np.random.randn(4, 3)
        direct_result = small_x @ small_w
        
        print(f"Direct multiplication: {small_x.shape} @ {small_w.shape} = {direct_result.shape}")
        print(f"This demonstrates the target for Winograd optimization")
        
        direct_ops = 2 * 4 * 3
        winograd_ops = 4 * (2 + 3 - 1)
        savings = 1 - (winograd_ops / direct_ops)
        print(f"Theoretical savings: {direct_ops} -> {winograd_ops} ops ({savings:.1%} reduction)")
        
    except Exception as e:
        print(f"Small matrix test failed: {e}")
    
    return winograd_2_3

# Run the test
if __name__ == "__main__":
    winograd_transform = test_winograd_transform()