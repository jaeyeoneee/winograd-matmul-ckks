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
        [4, 0, -5, 0, 1, 0],
        [0, -4, -4, 1, 1,0],
        [0, 4, -4, -1, 1,0],
        [0,-2, -1,  2, 1,0],
        [0, 2, -1, -2, 1,0],
        [0, 4,  0, -5, 0,1]
        ], dtype=np.float64)
        
        G = np.array([
            [1/4,     0,     0],
            [-1/6,   -1/6,   -1/6],
            [-1/6,  1/6,   -1/6],
            [1/24,   1/12,   1/6],
            [1/24,  -1/12,   1/6],
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
