"""
Chan-ZKP Protokol Aktörleri Modülü

Bu modül, Zero-Knowledge Proof protokolünün iki tarafını simüle eder:
    - Prover (Kanıtlayıcı): Gizli bilginin sahibi
    - Verifier (Doğrulayıcı): Gizli bilginin varlığını doğrulayan taraf

Protokol Akışı:
    1. Prover, Yeşil bir vektör (gizli bilgi) üretir
    2. Prover, bu vektörün commitment'ını (hash) gönderir
    3. Verifier, rastgele bir challenge (matris B) gönderir
    4. Prover, B*v işlemini yapar ve sonucu (w) gönderir
    5. Verifier, w'nin Mavi olduğunu ve matematiksel tutarlılığı doğrular
"""

import hashlib
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from .core import ColorOracle, MathEngine, Color


@dataclass
class Commitment:
    """Kanıtlayıcının taahhüdü (commitment)"""
    hash_value: str  # Gizli vektörün SHA-256 hash'i
    vector_dimension: int
    field_modulus: int


@dataclass
class Challenge:
    """Doğrulayıcının zorluğu (challenge)"""
    matrix_B: np.ndarray
    challenge_id: str  # Benzersiz zorluk kimliği


@dataclass 
class Response:
    """Kanıtlayıcının cevabı (response)"""
    w_vector: Optional[np.ndarray] = None  # Unblinded w = B * v (may be omitted in blinded mode)
    original_v: Optional[np.ndarray] = None  # Demo mode can show v
    blinded_w_vector: Optional[np.ndarray] = None  # Blinded response r * w (mod p)
    blinding_factor: Optional[int] = None  # r used for blinding (non-zero mod p)


@dataclass
class VerificationResult:
    """Doğrulama sonucu"""
    is_valid: bool
    details: Dict[str, Any]


class Prover:
    """
    Kanıtlayıcı (Prover)
    
    Gizli bir Yeşil vektöre sahip olan ve bu vektörün varlığını
    Doğrulayıcı'ya kanıtlamak isteyen taraf.
    
    In the context of Melody Chan's Theorem:
        - Green vector: Secret information (witness)
        - Matrix transformation: Proof mechanism
        - Blue result: Successful proof
    """
    
    def __init__(self, math_engine: MathEngine, verbose: bool = True):
        """
        Prover initializer.
        
        Args:
            math_engine: MathEngine instance for mathematical operations
            verbose: Detailed log output
        """
        self.engine = math_engine
        self.verbose = verbose
        self.secret_v: Optional[np.ndarray] = None
        self._commitment: Optional[Commitment] = None
        
    def _log(self, message: str):
        """Prints message in verbose mode."""
        if self.verbose:
            print(f"  [PROVER] {message}")
    
    def generate_secret(self, max_attempts: int = 10000) -> np.ndarray:
        """
        Generates a GREEN secret vector.
        
        This is a "mining" process - vectors are tried until
        the hash function produces a GREEN color.
        
        Args:
            max_attempts: Maximum number of attempts
            
        Returns:
            GREEN colored vector
            
        Raises:
            RuntimeError: If GREEN vector cannot be found
        """
        self._log("Searching for GREEN vector (mining)...")
        
        for attempt in range(1, max_attempts + 1):
            candidate = self.engine.random_vector()
            
            if ColorOracle.is_green(candidate):
                self.secret_v = candidate
                self._log(f"✓ GREEN vector found! (Attempt: {attempt})")
                self._log(f"  v = {candidate}")
                return candidate
        
        raise RuntimeError(f"GREEN vector not found ({max_attempts} attempts)")
    
    def commit(self) -> Commitment:
        """
        Creates a commitment for the secret vector.
        
        Commitment = SHA-256(v)
        
        This value proves the existence of the secret vector without revealing it.
        
        Returns:
            Commitment object
            
        Raises:
            ValueError: If secret vector has not been generated yet
        """
        if self.secret_v is None:
            raise ValueError("generate_secret() must be called first!")
        
        # Hash the vector
        vec_bytes = self.secret_v.astype(np.int64).tobytes()
        hash_value = hashlib.sha256(vec_bytes).hexdigest()
        
        self._commitment = Commitment(
            hash_value=hash_value,
            vector_dimension=self.engine.n,
            field_modulus=self.engine.p
        )
        
        self._log(f"Commitment created:")
        self._log(f"  Hash: {hash_value[:32]}...")
        
        return self._commitment
    
    def solve_challenge(self, challenge: Challenge) -> Response:
        """
        Solves the verifier's challenge.
        
        Operation: w = B * v (mod p)
        
        Chan Theorem guarantee: If v is GREEN and B is an appropriate matrix,
        the theorem guarantees that w will be BLUE under certain conditions.
        
        Args:
            challenge: Challenge object sent by the verifier
            
        Returns:
            Response object (including w vector)
        """
        if self.secret_v is None:
            raise ValueError("generate_secret() must be called first!")
        
        B = challenge.matrix_B
        v = self.secret_v
        
        self._log(f"Challenge received (ID: {challenge.challenge_id})")
        self._log(f"  Matrix B size: {B.shape}")
        
        # w = B * v (mod p)
        w = self.engine.matrix_vector_mult(B, v)

        # Blinding: choose random non-zero factor r and blind w
        r = int(np.random.randint(1, self.engine.p))  # 1..p-1
        blinded_w = (r * w) % self.engine.p
        
        # Check the color of the result
        w_color = ColorOracle.get_color_name(w)
        v_color = ColorOracle.get_color_name(v)
        
        self._log(f"Multiplication calculated:")
        self._log(f"  v = {v} ({v_color})")
        self._log(f"  w = B*v = {w} ({w_color})")
        self._log(f"  Blinding factor r = {r}")
        self._log(f"  Blinded w = r*w mod p = {blinded_w}")
        
        if ColorOracle.is_blue(w):
            self._log(f"✓ Success! w is BLUE - Proof is valid!")
        else:
            self._log(f"⚠ Warning: w is GREEN - Proof is weak for this challenge")
        
        return Response(
            w_vector=w,
            original_v=v,  # We also return v in demo mode
            blinded_w_vector=blinded_w,
            blinding_factor=r
        )
    
    def find_valid_green_for_challenge(self, challenge: Challenge, 
                                       max_attempts: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds a GREEN vector v for the given challenge such that w=B*v results in BLUE.
        
        This is the practical application of Chan's Theorem:
        The theorem guarantees that such a v EXISTS.
        
        Args:
            challenge: Verifier's challenge
            max_attempts: Maximum number of attempts
            
        Returns:
            (v, w) tuple - GREEN v and BLUE w
        """
        B = challenge.matrix_B
        self._log(f"Chan Theorem verification: Searching for GREEN->BLUE transformation...")
        
        for attempt in range(1, max_attempts + 1):
            # Generate random vector
            v = self.engine.random_vector()
            
            # Is v GREEN?
            if not ColorOracle.is_green(v):
                continue
            
            # Calculate w = B * v
            w = self.engine.matrix_vector_mult(B, v)
            
            # Is w BLUE?
            if ColorOracle.is_blue(w):
                self._log(f"✓ Chan pair found! (Attempt: {attempt})")
                self._log(f"  GREEN v = {v}")
                self._log(f"  BLUE  w = {w}")
                self.secret_v = v
                return v, w
        
        raise RuntimeError(f"GREEN->BLUE pair not found ({max_attempts} attempts)")


class Verifier:
    """
    Verifier
    
    The party that wants to verify that the prover truly possesses
    a GREEN vector.
    
    The verifier NEVER sees the secret vector (v), only:
        - Commitment (hash)
        - Response (w = B*v)
    values.
    """
    
    def __init__(self, math_engine: MathEngine, verbose: bool = True):
        """
        Verifier initializer.
        
        Args:
            math_engine: MathEngine instance for mathematical operations
            verbose: Detailed log output
        """
        self.engine = math_engine
        self.verbose = verbose
        self._current_challenge: Optional[Challenge] = None
        
    def _log(self, message: str):
        """Prints message in verbose mode."""
        if self.verbose:
            print(f"  [VERIFIER] {message}")
    
    def generate_challenge(self) -> Challenge:
        """
        Generates a challenge to send to the prover.
        
        Challenge = Random non-singular matrix B (excluding identity matrix)
        
        Chan Theorem condition: B must not be singular and must not be identity.
        
        Returns:
            Challenge object
        """
        self._log("Generating challenge...")
        
        # Non-singular, non-identity matrix
        B = self.engine.random_nonsingular_matrix(exclude_identity=True)
        
        # Unique challenge ID
        challenge_id = hashlib.sha256(B.tobytes()).hexdigest()[:8]
        
        self._current_challenge = Challenge(
            matrix_B=B,
            challenge_id=challenge_id
        )
        
        self._log(f"Challenge created (ID: {challenge_id})")
        self._log(f"  Matrix B:\n{B}")
        self._log(f"  det(B) mod {self.engine.p} = {self.engine._determinant_mod_p(B)}")
        
        return self._current_challenge
    
    def verify(self, commitment: Commitment, response: Response, 
               reveal_v: bool = False) -> VerificationResult:
        """
        Verifies the prover's response.
        
        Verification steps:
            1. Check if w vector is BLUE
            2. (Optional) If v is provided, verify commitment hash
            3. (Optional) If v is provided, check B*v = w equality
        
        Args:
            commitment: Commitment initially sent by the prover
            response: Prover's response to the challenge
            reveal_v: True if v is revealed scenario (full verification)
            
        Returns:
            VerificationResult object
        """
        if self._current_challenge is None:
            raise ValueError("generate_challenge() must be called first!")
        
        self._log("Verification starting...")
        
        B = self._current_challenge.matrix_B

        # Unblind if needed
        if response.blinded_w_vector is not None and response.blinding_factor is not None:
            r = int(response.blinding_factor)
            r_inv = self.engine._mod_inverse(r, self.engine.p)
            if r_inv is None:
                raise ValueError("Invalid blinding factor; inverse does not exist.")
            w = (response.blinded_w_vector.astype(np.int64) * r_inv) % self.engine.p
            w = w.astype(np.int64)
            details_source = "blinded"
        elif response.w_vector is not None:
            w = response.w_vector.astype(np.int64)
            details_source = "unblinded"
        else:
            raise ValueError("No response vector provided.")
        
        details = {
            "challenge_id": self._current_challenge.challenge_id,
            "w_vector": w.tolist(),
            "response_source": details_source,
            "checks_passed": []
        }
        
        # CHECK 1: Is w BLUE?
        w_is_blue = ColorOracle.is_blue(w)
        details["w_is_blue"] = w_is_blue
        w_color = ColorOracle.get_color_name(w)
        
        self._log(f"[CHECK 1] w color: {w_color}")
        
        if w_is_blue:
            self._log(f"  ✓ w is BLUE - Chan Theorem condition satisfied!")
            details["checks_passed"].append("w_is_blue")
        else:
            self._log(f"  ✗ w is GREEN - Proof failed!")
        
        # If v is revealed (demo/reveal mode)
        if reveal_v and response.original_v is not None:
            v = response.original_v
            
            # CHECK 2: Is v GREEN?
            v_is_green = ColorOracle.is_green(v)
            details["v_is_green"] = v_is_green
            v_color = ColorOracle.get_color_name(v)
            
            self._log(f"[CHECK 2] v color: {v_color}")
            
            if v_is_green:
                self._log(f"  ✓ v is GREEN - Valid secret vector!")
                details["checks_passed"].append("v_is_green")
            else:
                self._log(f"  ✗ v is BLUE - Invalid secret vector!")
            
            # CHECK 3: Commitment verification
            v_bytes = v.astype(np.int64).tobytes()
            v_hash = hashlib.sha256(v_bytes).hexdigest()
            commitment_valid = (v_hash == commitment.hash_value)
            details["commitment_valid"] = commitment_valid
            
            self._log(f"[CHECK 3] Commitment verification:")
            
            if commitment_valid:
                self._log(f"  ✓ Hash matched - v unchanged!")
                details["checks_passed"].append("commitment_valid")
            else:
                self._log(f"  ✗ Hash mismatch - v may have been changed!")
            
            # CHECK 4: B*v = w equality
            calculated_w = self.engine.matrix_vector_mult(B, v)
            math_valid = np.array_equal(calculated_w, w)
            details["math_valid"] = math_valid
            
            self._log(f"[CHECK 4] Mathematical verification (B*v = w):")
            
            if math_valid:
                self._log(f"  ✓ B*v = {calculated_w} = w ✓")
                details["checks_passed"].append("math_valid")
            else:
                self._log(f"  ✗ B*v = {calculated_w} ≠ w = {w}")
        
        # Result evaluation
        # In ZKP mode, only w being BLUE is sufficient
        # In reveal mode, all checks must pass
        if reveal_v:
            is_valid = len(details["checks_passed"]) == 4
        else:
            is_valid = w_is_blue
        
        details["final_result"] = "SUCCESSFUL" if is_valid else "FAILED"
        
        self._log("-" * 40)
        if is_valid:
            self._log(f"✓✓✓ VERIFICATION SUCCESSFUL ✓✓✓")
        else:
            self._log(f"✗✗✗ VERIFICATION FAILED ✗✗✗")
        
        return VerificationResult(is_valid=is_valid, details=details)


def run_protocol_demo(dimension: int = 4, modulus: int = 7, verbose: bool = True):
    """
    Runs the full ZKP protocol demo.
    
    Args:
        dimension: Vector dimension (n)
        modulus: Field modulus (p)
        verbose: Detailed output
    """
    print("=" * 60)
    print("CHAN-ZKP PROTOCOL DEMO")
    print("=" * 60)
    print(f"\nParameters: n={dimension}, p={modulus} (GF({modulus}))")
    print("=" * 60)
    
    # Shared math engine
    engine = MathEngine(dimension=dimension, modulus=modulus)
    
    # Create actors
    prover = Prover(engine, verbose=verbose)
    verifier = Verifier(engine, verbose=verbose)
    
    # STEP 1: Prover generates secret vector
    print("\n[STEP 1] Prover: Secret GREEN vector generation")
    print("-" * 40)
    try:
        v = prover.generate_secret()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return
    
    # STEP 2: Prover sends commitment
    print("\n[STEP 2] Prover -> Verifier: Commitment submission")
    print("-" * 40)
    commitment = prover.commit()
    
    # STEP 3: Verifier sends challenge
    print("\n[STEP 3] Verifier -> Prover: Challenge submission")
    print("-" * 40)
    challenge = verifier.generate_challenge()
    
    # STEP 4: Prover solves challenge and sends response
    print("\n[STEP 4] Prover: Challenge solution and Response submission")
    print("-" * 40)
    response = prover.solve_challenge(challenge)
    
    # STEP 5: Verifier performs verification
    print("\n[STEP 5] Verifier: Verification (in Reveal mode)")
    print("-" * 40)
    result = verifier.verify(commitment, response, reveal_v=True)
    
    # Result summary
    print("\n" + "=" * 60)
    print("PROTOCOL RESULT")
    print("=" * 60)
    print(f"Passed checks: {result.details['checks_passed']}")
    print(f"Final: {result.details['final_result']}")
    
    return result


# Test için çalıştır
if __name__ == "__main__":
    run_protocol_demo(dimension=4, modulus=7)

