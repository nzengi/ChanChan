"""
Chan-ZKP Matematiksel Çekirdek Modülü

Bu modül, Melody Chan teoreminin kriptografik uygulaması için
gerekli matematiksel altyapıyı sağlar.

Sınıflar:
    - ColorOracle: Vektörleri deterministik olarak Yeşil/Mavi olarak renklendirir
    - MathEngine: Galois Field üzerinde matris/vektör işlemleri yapar
"""

import hashlib
import hmac
import os
import numpy as np
from typing import Tuple, Optional
from enum import Enum


class Color(Enum):
    """Vektör renk tanımları (Chan Teoremi)"""
    GREEN = 0  # Yeşil - Gizli vektörler bu renkte olmalı
    BLUE = 1   # Mavi - Dönüşüm sonucu bu renk olmalı


DEFAULT_COLOR_KEY = b"chan-zkp-color-key"
COLOR_TAG = b"CHAN-ZKP-COLOR"


class ColorOracle:
    """
    Renk Kehaneti (Color Oracle)
    
    Bir vektörü SHA-256 hash fonksiyonu kullanarak
    deterministik olarak Yeşil veya Mavi olarak etiketler.
    
    Chan Teoremi Bağlamı:
        - Yeşil vektörler: Kanıtlayıcının seçeceği gizli vektörler
        - Mavi vektörler: Matris dönüşümü sonrası hedef vektörler
    """
    
    @staticmethod
    def get_color(vector: np.ndarray, key: bytes = None) -> Color:
        """
        Bir vektörün rengini belirler.
        
        Algoritma:
            1. Vektörü byte dizisine çevir
            2. SHA-256 ile hashle
            3. Hash'in son byte'ının paritesine göre renk belirle
        
        Args:
            vector: numpy ndarray formatında vektör
            
        Returns:
            Color.GREEN veya Color.BLUE
        """
        # Vektörü int64 formatına çevir (tutarlılık için)
        vec_normalized = vector.astype(np.int64)
        vec_bytes = vec_normalized.tobytes()
        
        # Keyed HMAC-SHA256 (daha güvenli ve dengeli)
        key = key or os.getenv("CHAN_ZKP_COLOR_KEY", DEFAULT_COLOR_KEY)
        key_bytes = key if isinstance(key, (bytes, bytearray)) else str(key).encode()
        # Domain separation/tagging for coloring
        hash_digest = hmac.new(key_bytes, COLOR_TAG + vec_bytes, hashlib.sha256).digest()
        
        # İlk byte'ın paritesine göre renk (daha dengeli, keyed)
        first_byte = hash_digest[0]
        
        if first_byte % 2 == 0:
            return Color.GREEN
        else:
            return Color.BLUE
    
    @staticmethod
    def is_green(vector: np.ndarray) -> bool:
        """Vektörün Yeşil olup olmadığını kontrol eder."""
        return ColorOracle.get_color(vector) == Color.GREEN
    
    @staticmethod
    def is_blue(vector: np.ndarray) -> bool:
        """Vektörün Mavi olup olmadığını kontrol eder."""
        return ColorOracle.get_color(vector) == Color.BLUE
    
    @staticmethod
    def get_color_name(vector: np.ndarray) -> str:
        """Returns the vector's color as an English string."""
        color = ColorOracle.get_color(vector)
        return "GREEN" if color == Color.GREEN else "BLUE"


class MathEngine:
    """
    Matematik Motoru
    
    Galois Field (Sonlu Alan) GF(p) üzerinde matris ve vektör
    işlemlerini gerçekleştirir.
    
    Chan Teoremi için kritik özellikler:
        - Modüler aritmetik (tüm işlemler mod p)
        - Non-singular (tekil olmayan) matris üretimi
        - Matris tersi hesaplama
    """
    
    def __init__(self, dimension: int, modulus: int):
        """
        MathEngine başlatıcı.
        
        Args:
            dimension: Vektör boyutu (n)
            modulus: Alan modülü (p) - Asal sayı olmalı
            
        Chan Teoremi Koşulu:
            |F| > n + 1, yani modulus > dimension + 1
        """
        if modulus <= dimension + 1:
            raise ValueError(
                f"Chan Teoremi koşulu: modulus ({modulus}) > dimension + 1 ({dimension + 1}) olmalı!"
            )
        
        self.n = dimension
        self.p = modulus
        
    def random_vector(self) -> np.ndarray:
        """GF(p) üzerinde rastgele bir n-vektör üretir."""
        return np.random.randint(0, self.p, self.n)
    
    def random_nonsingular_matrix(self, exclude_identity: bool = True) -> np.ndarray:
        """
        GF(p) üzerinde tekil olmayan (non-singular) bir n×n matris üretir.
        
        Args:
            exclude_identity: True ise birim matris hariç tutulur (Chan Teoremi gereksinimi)
            
        Returns:
            Determinantı 0 olmayan (mod p) matris
        """
        max_attempts = 1000
        
        for _ in range(max_attempts):
            matrix = np.random.randint(0, self.p, (self.n, self.n))
            
            # Birim matris kontrolü
            if exclude_identity and np.array_equal(matrix % self.p, np.eye(self.n) % self.p):
                continue
            
            # Determinant kontrolü (mod p)
            det = self._determinant_mod_p(matrix)
            if det != 0:
                return matrix
        
        raise RuntimeError("Non-singular matris üretilemedi!")
    
    def _determinant_mod_p(self, matrix: np.ndarray) -> int:
        """
        Matrisin determinantını mod p hesaplar.
        
        Not: numpy.linalg.det float döndürür ve büyük sayılarda
        hassasiyet kaybeder. Bu yüzden özel implementasyon kullanıyoruz.
        """
        n = len(matrix)
        mat = matrix.astype(np.int64) % self.p
        
        # LU dekompozisyonu benzeri yaklaşım
        det = 1
        for col in range(n):
            # Pivot bulma
            pivot_row = None
            for row in range(col, n):
                if mat[row, col] % self.p != 0:
                    pivot_row = row
                    break
            
            if pivot_row is None:
                return 0  # Matris singular
            
            # Satır değişimi
            if pivot_row != col:
                mat[[col, pivot_row]] = mat[[pivot_row, col]]
                det = (-det) % self.p
            
            # Pivot elemanı
            pivot = mat[col, col] % self.p
            det = (det * pivot) % self.p
            
            # Eliminasyon
            pivot_inv = self._mod_inverse(pivot, self.p)
            if pivot_inv is None:
                return 0
                
            for row in range(col + 1, n):
                factor = (mat[row, col] * pivot_inv) % self.p
                mat[row] = (mat[row] - factor * mat[col]) % self.p
        
        return det % self.p
    
    def _mod_inverse(self, a: int, p: int) -> Optional[int]:
        """
        a'nın mod p tersini hesaplar (Extended Euclidean Algorithm).
        
        Returns:
            a^(-1) mod p veya None (ters yoksa)
        """
        a = a % p
        if a == 0:
            return None
            
        # Fermat'ın Küçük Teoremi: a^(p-1) ≡ 1 (mod p) için asal p
        # Dolayısıyla: a^(-1) ≡ a^(p-2) (mod p)
        return pow(int(a), p - 2, p)
    
    def matrix_inverse_mod_p(self, matrix: np.ndarray) -> Optional[np.ndarray]:
        """
        Matrisin mod p tersini hesaplar.
        
        Returns:
            Ters matris veya None (ters yoksa)
        """
        n = len(matrix)
        
        # Augmented matris [A | I]
        aug = np.zeros((n, 2 * n), dtype=np.int64)
        aug[:, :n] = matrix.astype(np.int64) % self.p
        aug[:, n:] = np.eye(n, dtype=np.int64)
        
        # Gauss-Jordan eliminasyonu
        for col in range(n):
            # Pivot bulma
            pivot_row = None
            for row in range(col, n):
                if aug[row, col] % self.p != 0:
                    pivot_row = row
                    break
            
            if pivot_row is None:
                return None  # Matris singular
            
            # Satır değişimi
            if pivot_row != col:
                aug[[col, pivot_row]] = aug[[pivot_row, col]]
            
            # Pivot'u 1 yap
            pivot = aug[col, col] % self.p
            pivot_inv = self._mod_inverse(int(pivot), self.p)
            if pivot_inv is None:
                return None
            
            aug[col] = (aug[col] * pivot_inv) % self.p
            
            # Diğer satırları sıfırla
            for row in range(n):
                if row != col:
                    factor = aug[row, col] % self.p
                    aug[row] = (aug[row] - factor * aug[col]) % self.p
        
        # Ters matris sağ yarıda
        inverse = aug[:, n:] % self.p
        return inverse.astype(np.int64)
    
    def matrix_vector_mult(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Matris-vektör çarpımı (mod p).
        
        w = B * v (mod p)
        """
        result = np.dot(matrix.astype(np.int64), vector.astype(np.int64))
        return (result % self.p).astype(np.int64)
    
    def verify_nonsingular(self, matrix: np.ndarray) -> bool:
        """Matrisin tekil olmadığını doğrular."""
        det = self._determinant_mod_p(matrix)
        return det != 0
    
    def is_identity(self, matrix: np.ndarray) -> bool:
        """Matrisin birim matris olup olmadığını kontrol eder."""
        identity = np.eye(self.n, dtype=np.int64)
        return np.array_equal(matrix.astype(np.int64) % self.p, identity)


# Test için basit bir demo
if __name__ == "__main__":
    print("=" * 60)
    print("Chan-ZKP Matematik Motoru Demo")
    print("=" * 60)
    
    # Parametreler: n=4, p=7 (Chan'ın orijinal örneği)
    engine = MathEngine(dimension=4, modulus=7)
    
    print(f"\nParametreler: n={engine.n}, p={engine.p} (GF({engine.p}))")
    print("-" * 40)
    
    # Rastgele vektör üret ve rengini kontrol et
    print("\n[1] Rastgele Vektör Üretimi ve Renklendirme:")
    for i in range(5):
        v = engine.random_vector()
        color = ColorOracle.get_color_name(v)
        print(f"    v{i+1} = {v} -> {color}")
    
    # Non-singular matris üret
    print("\n[2] Non-singular Matris Üretimi:")
    B = engine.random_nonsingular_matrix()
    print(f"    B =\n{B}")
    print(f"    det(B) mod {engine.p} = {engine._determinant_mod_p(B)}")
    
    # Matris-vektör çarpımı
    print("\n[3] Matris-Vektör Çarpımı:")
    v = engine.random_vector()
    w = engine.matrix_vector_mult(B, v)
    print(f"    v = {v} ({ColorOracle.get_color_name(v)})")
    print(f"    w = B*v = {w} ({ColorOracle.get_color_name(w)})")
    
    print("\n" + "=" * 60)
    print("Demo tamamlandı!")

