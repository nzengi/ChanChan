"""
Chan-ZKP Mathematical Core Module

This module provides the mathematical foundation for the cryptographic
application of Melody Chan's theorem.

Classes:
    - ColorOracle: Deterministically colors vectors as Green/Blue
    - MathEngine: Performs matrix/vector operations over Galois Field
"""

import hashlib
import hmac
import numpy as np
from typing import Tuple, Optional, Union
from enum import Enum

from .config import get_config

class Color(Enum):
    """Vector color definitions (Chan's Theorem)"""
    GREEN = 0  # Green - Secret vectors must be this color
    BLUE = 1   # Blue - Transformation result should be this color


COLOR_TAG = b"CHAN-ZKP-COLOR"


class ColorOracle:
    """
    Color Oracle
    
    Deterministically labels a vector as Green or Blue using
    a keyed HMAC-SHA256 hash function.
    
    Chan's Theorem Context:
        - Green vectors: Secret vectors chosen by the prover
        - Blue vectors: Target vectors after matrix transformation
    """
    
    @staticmethod
    def get_color(vector: np.ndarray, key: Optional[Union[bytes, str]] = None) -> Color:
        """
        Determines the color of a vector.
        
        Algorithm:
            1. Convert vector to byte array
            2. Hash with keyed HMAC-SHA256
            3. Determine color based on first byte parity
        
        Args:
            vector: Vector in numpy ndarray format
            key: Optional key for HMAC (defaults to env var or default)
            
        Returns:
            Color.GREEN or Color.BLUE
        """
        # Convert vector to int64 format (for consistency)
        vec_normalized = vector.astype(np.int64)
        vec_bytes = vec_normalized.tobytes()
        
        # Keyed HMAC-SHA256 (more secure and balanced)
        if key is None:
            config = get_config()
            key = config.get_color_key()
        key_bytes = key if isinstance(key, (bytes, bytearray)) else str(key).encode()
        # Domain separation/tagging for coloring
        hash_digest = hmac.new(key_bytes, COLOR_TAG + vec_bytes, hashlib.sha256).digest()
        
        # Color based on first byte parity (more balanced, keyed)
        first_byte = hash_digest[0]
        
        if first_byte % 2 == 0:
            return Color.GREEN
        else:
            return Color.BLUE
    
    @staticmethod
    def is_green(vector: np.ndarray) -> bool:
        """Checks if the vector is Green."""
        return ColorOracle.get_color(vector) == Color.GREEN
    
    @staticmethod
    def is_blue(vector: np.ndarray) -> bool:
        """Checks if the vector is Blue."""
        return ColorOracle.get_color(vector) == Color.BLUE
    
    @staticmethod
    def get_color_name(vector: np.ndarray) -> str:
        """Returns the vector's color as an English string."""
        color = ColorOracle.get_color(vector)
        return "GREEN" if color == Color.GREEN else "BLUE"


class MathEngine:
    """
    Mathematical Engine
    
    Performs matrix and vector operations over Galois Field (Finite Field) GF(p).
    
    Critical features for Chan's Theorem:
        - Modular arithmetic (all operations mod p)
        - Non-singular matrix generation
        - Matrix inverse computation
    """
    
    def __init__(self, dimension: int, modulus: int):
        """
        MathEngine initializer.
        
        Args:
            dimension: Vector dimension (n)
            modulus: Field modulus (p) - must be prime
            
        Chan's Theorem Condition:
            |F| > n + 1, i.e., modulus > dimension + 1
        """
        if modulus <= dimension + 1:
            raise ValueError(
                f"Chan's Theorem condition: modulus ({modulus}) must be > dimension + 1 ({dimension + 1})!"
            )
        
        self.n = dimension
        self.p = modulus
        
    def random_vector(self) -> np.ndarray:
        """Generates a random n-vector over GF(p)."""
        return np.random.randint(0, self.p, self.n)
    
    def random_nonsingular_matrix(self, exclude_identity: bool = True) -> np.ndarray:
        """
        Generates a non-singular n×n matrix over GF(p).
        
        Args:
            exclude_identity: If True, identity matrix is excluded (Chan's Theorem requirement)
            
        Returns:
            Matrix with non-zero determinant (mod p)
        """
        max_attempts = 1000
        
        for _ in range(max_attempts):
            matrix = np.random.randint(0, self.p, (self.n, self.n))
            
            # Identity matrix check
            if exclude_identity and np.array_equal(matrix % self.p, np.eye(self.n) % self.p):
                continue
            
            # Determinant check (mod p)
            det = self._determinant_mod_p(matrix)
            if det != 0:
                return matrix
        
        raise RuntimeError("Failed to generate non-singular matrix!")
    
    def _determinant_mod_p(self, matrix: np.ndarray) -> int:
        """
        Computes the determinant of a matrix mod p.
        
        Note: numpy.linalg.det returns float and loses precision
        for large numbers. Therefore, we use a custom implementation.
        """
        n = len(matrix)
        mat = matrix.astype(np.int64) % self.p
        
        # LU decomposition-like approach
        det = 1
        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(col, n):
                if mat[row, col] % self.p != 0:
                    pivot_row = row
                    break
            
            if pivot_row is None:
                return 0  # Matrix is singular
            
            # Row swap
            if pivot_row != col:
                mat[[col, pivot_row]] = mat[[pivot_row, col]]
                det = (-det) % self.p
            
            # Pivot element
            pivot = mat[col, col] % self.p
            det = (det * pivot) % self.p
            
            # Elimination
            pivot_inv = self._mod_inverse(pivot, self.p)
            if pivot_inv is None:
                return 0
                
            for row in range(col + 1, n):
                factor = (mat[row, col] * pivot_inv) % self.p
                mat[row] = (mat[row] - factor * mat[col]) % self.p
        
        return det % self.p
    
    def _mod_inverse(self, a: int, p: int) -> Optional[int]:
        """
        Computes the modular inverse of a mod p (using Fermat's Little Theorem).
        
        Args:
            a: Number to invert
            p: Prime modulus
            
        Returns:
            a^(-1) mod p or None if inverse doesn't exist
        """
        a = a % p
        if a == 0:
            return None
            
        # Fermat's Little Theorem: a^(p-1) ≡ 1 (mod p) for prime p
        # Therefore: a^(-1) ≡ a^(p-2) (mod p)
        return pow(int(a), p - 2, p)
    
    def matrix_inverse_mod_p(self, matrix: np.ndarray) -> Optional[np.ndarray]:
        """
        Computes the modular inverse of a matrix mod p.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Inverse matrix or None if inverse doesn't exist
        """
        n = len(matrix)
        
        # Augmented matrix [A | I]
        aug = np.zeros((n, 2 * n), dtype=np.int64)
        aug[:, :n] = matrix.astype(np.int64) % self.p
        aug[:, n:] = np.eye(n, dtype=np.int64)
        
        # Gauss-Jordan elimination
        for col in range(n):
            # Find pivot
            pivot_row: Optional[int] = None
            for row in range(col, n):
                if aug[row, col] % self.p != 0:
                    pivot_row = row
                    break
            
            if pivot_row is None:
                return None  # Matrix is singular
            
            # Row swap
            if pivot_row != col:
                aug[[col, pivot_row]] = aug[[pivot_row, col]]
            
            # Make pivot 1
            pivot = aug[col, col] % self.p
            pivot_inv = self._mod_inverse(int(pivot), self.p)
            if pivot_inv is None:
                return None
            
            aug[col] = (aug[col] * pivot_inv) % self.p
            
            # Zero out other rows
            for row in range(n):
                if row != col:
                    factor = aug[row, col] % self.p
                    aug[row] = (aug[row] - factor * aug[col]) % self.p
        
        # Inverse matrix is in the right half
        inverse = aug[:, n:] % self.p
        return inverse.astype(np.int64)
    
    def matrix_vector_mult(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Matrix-vector multiplication (mod p).
        
        Args:
            matrix: Matrix B
            vector: Vector v
            
        Returns:
            w = B * v (mod p)
        """
        result = np.dot(matrix.astype(np.int64), vector.astype(np.int64))
        return (result % self.p).astype(np.int64)
    
    def verify_nonsingular(self, matrix: np.ndarray) -> bool:
        """Verifies that the matrix is non-singular."""
        det = self._determinant_mod_p(matrix)
        return det != 0
    
    def is_identity(self, matrix: np.ndarray) -> bool:
        """Checks if the matrix is the identity matrix."""
        identity = np.eye(self.n, dtype=np.int64)
        return np.array_equal(matrix.astype(np.int64) % self.p, identity)


# Simple demo for testing
if __name__ == "__main__":
    print("=" * 60)
    print("Chan-ZKP Mathematical Engine Demo")
    print("=" * 60)
    
    # Parameters: n=4, p=7 (Chan's original example)
    engine = MathEngine(dimension=4, modulus=7)
    
    print(f"\nParameters: n={engine.n}, p={engine.p} (GF({engine.p}))")
    print("-" * 40)
    
    # Generate random vector and check its color
    print("\n[1] Random Vector Generation and Coloring:")
    for i in range(5):
        v = engine.random_vector()
        color = ColorOracle.get_color_name(v)
        print(f"    v{i+1} = {v} -> {color}")
    
    # Generate non-singular matrix
    print("\n[2] Non-singular Matrix Generation:")
    B = engine.random_nonsingular_matrix()
    print(f"    B =\n{B}")
    print(f"    det(B) mod {engine.p} = {engine._determinant_mod_p(B)}")
    
    # Matrix-vector multiplication
    print("\n[3] Matrix-Vector Multiplication:")
    v = engine.random_vector()
    w = engine.matrix_vector_mult(B, v)
    print(f"    v = {v} ({ColorOracle.get_color_name(v)})")
    print(f"    w = B*v = {w} ({ColorOracle.get_color_name(w)})")
    
    print("\n" + "=" * 60)
    print("Demo completed!")

