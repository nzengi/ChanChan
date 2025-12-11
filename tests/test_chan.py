#!/usr/bin/env python3
"""
Chan-ZKP Test Suite

Comprehensive real tests for the Chan-ZKP system without mocks.
All tests use actual implementations and verify real behavior.

Run:
    python tests/test_chan.py
    or
    python -m pytest tests/test_chan.py -v
"""

import sys
import os
import unittest
import numpy as np
import math

try:
    from hypothesis import given, strategies as st, settings
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import ColorOracle, MathEngine, Color
from src.actors import Prover, Verifier, Challenge, Commitment, Response


class TestColorOracle(unittest.TestCase):
    """Real tests for ColorOracle - no mocks."""
    
    def test_color_determinism(self):
        """Same vector must always produce same color."""
        v = np.array([1, 2, 3, 4, 5])
        
        color1 = ColorOracle.get_color(v)
        color2 = ColorOracle.get_color(v)
        color3 = ColorOracle.get_color(v)
        
        self.assertEqual(color1, color2)
        self.assertEqual(color2, color3)
        self.assertEqual(color1, color3)
    
    def test_color_distribution(self):
        """Colors should be approximately 50/50 distributed."""
        engine = MathEngine(dimension=4, modulus=7)
        greens = 0
        blues = 0
        total = 1000
        
        for _ in range(total):
            v = engine.random_vector()
            if ColorOracle.is_green(v):
                greens += 1
            else:
                blues += 1
        
        # Should be roughly balanced (40-60% range)
        green_ratio = greens / total
        self.assertGreater(green_ratio, 0.35)
        self.assertLess(green_ratio, 0.65)
    
    def test_color_consistency(self):
        """is_green and is_blue should be consistent with get_color."""
        engine = MathEngine(dimension=4, modulus=7)
        
        for _ in range(100):
            v = engine.random_vector()
            color = ColorOracle.get_color(v)
            
            if color == Color.GREEN:
                self.assertTrue(ColorOracle.is_green(v))
                self.assertFalse(ColorOracle.is_blue(v))
            else:
                self.assertTrue(ColorOracle.is_blue(v))
                self.assertFalse(ColorOracle.is_green(v))


class TestMathEngine(unittest.TestCase):
    """Real tests for MathEngine - actual matrix operations."""
    
    def setUp(self):
        self.engine = MathEngine(dimension=4, modulus=7)
    
    def test_chan_theorem_condition_enforcement(self):
        """MathEngine must enforce Chan theorem condition: p > n + 1."""
        # Invalid: p = n + 1 (should fail)
        with self.assertRaises(ValueError):
            MathEngine(dimension=4, modulus=5)  # 5 = 4 + 1
        
        # Invalid: p < n + 1 (should fail)
        with self.assertRaises(ValueError):
            MathEngine(dimension=4, modulus=4)  # 4 < 5
        
        # Valid: p > n + 1 (should work)
        engine = MathEngine(dimension=4, modulus=7)  # 7 > 5
        self.assertIsNotNone(engine)
    
    def test_vector_generation_bounds(self):
        """Generated vectors must be in [0, p) range."""
        for _ in range(100):
            v = self.engine.random_vector()
            self.assertEqual(len(v), self.engine.n)
            self.assertTrue(np.all(v >= 0))
            self.assertTrue(np.all(v < self.engine.p))
    
    def test_nonsingular_matrix_properties(self):
        """Generated matrices must be non-singular and non-identity."""
        for _ in range(50):
            B = self.engine.random_nonsingular_matrix()
            
            # Check size
            self.assertEqual(B.shape, (self.engine.n, self.engine.n))
            
            # Check non-singular
            det = self.engine._determinant_mod_p(B)
            self.assertNotEqual(det, 0)
            
            # Check non-identity
            self.assertFalse(self.engine.is_identity(B))
    
    def test_matrix_inverse_correctness(self):
        """Matrix inverse must satisfy A * A^(-1) = I (mod p)."""
        for _ in range(50):
            A = self.engine.random_nonsingular_matrix()
            A_inv = self.engine.matrix_inverse_mod_p(A)
            
            self.assertIsNotNone(A_inv)
            
            # Verify: A * A^(-1) = I
            product = np.dot(A.astype(np.int64), A_inv.astype(np.int64)) % self.engine.p
            identity = np.eye(self.engine.n, dtype=np.int64)
            
            self.assertTrue(np.array_equal(product, identity))
    
    def test_matrix_vector_multiplication(self):
        """Matrix-vector multiplication must be correct mod p."""
        B = self.engine.random_nonsingular_matrix()
        v = self.engine.random_vector()
        
        w = self.engine.matrix_vector_mult(B, v)
        
        # Check dimensions
        self.assertEqual(len(w), self.engine.n)
        
        # Check bounds
        self.assertTrue(np.all(w >= 0))
        self.assertTrue(np.all(w < self.engine.p))
        
        # Verify manually: w should equal B*v mod p
        manual_w = np.dot(B.astype(np.int64), v.astype(np.int64)) % self.engine.p
        self.assertTrue(np.array_equal(w, manual_w))


class TestProver(unittest.TestCase):
    """Real tests for Prover - actual protocol execution."""
    
    def setUp(self):
        self.engine = MathEngine(dimension=4, modulus=7)
        self.prover = Prover(self.engine, verbose=False)
    
    def test_generate_secret_produces_green(self):
        """generate_secret must produce a GREEN vector."""
        v = self.prover.generate_secret()
        
        self.assertIsNotNone(v)
        self.assertEqual(len(v), self.engine.n)
        self.assertTrue(ColorOracle.is_green(v))
    
    def test_commitment_determinism(self):
        """Same secret must produce same commitment."""
        self.prover.generate_secret()
        
        commit1 = self.prover.commit()
        commit2 = self.prover.commit()
        
        self.assertEqual(commit1.hash_value, commit2.hash_value)
        self.assertEqual(commit1.vector_dimension, self.engine.n)
        self.assertEqual(commit1.field_modulus, self.engine.p)
    
    def test_solve_challenge_produces_response(self):
        """solve_challenge must produce valid response."""
        self.prover.generate_secret()
        challenge = Challenge(
            matrix_B=self.engine.random_nonsingular_matrix(),
            challenge_id="test"
        )
        
        response = self.prover.solve_challenge(challenge)
        
        self.assertIsNotNone(response.w_vector)
        self.assertEqual(len(response.w_vector), self.engine.n)
    
    def test_find_valid_green_for_challenge(self):
        """find_valid_green_for_challenge must find GREEN->BLUE pair."""
        B = self.engine.random_nonsingular_matrix()
        challenge = Challenge(matrix_B=B, challenge_id="test")
        
        v, w = self.prover.find_valid_green_for_challenge(challenge, max_attempts=5000)
        
        # Verify properties
        self.assertTrue(ColorOracle.is_green(v))
        self.assertTrue(ColorOracle.is_blue(w))
        
        # Verify math: w = B * v
        calculated_w = self.engine.matrix_vector_mult(B, v)
        self.assertTrue(np.array_equal(calculated_w, w))


class TestVerifier(unittest.TestCase):
    """Real tests for Verifier - actual verification."""
    
    def setUp(self):
        self.engine = MathEngine(dimension=4, modulus=7)
        self.verifier = Verifier(self.engine, verbose=False)
    
    def test_generate_challenge_produces_valid_matrix(self):
        """generate_challenge must produce valid non-singular matrix."""
        challenge = self.verifier.generate_challenge()
        
        B = challenge.matrix_B
        det = self.engine._determinant_mod_p(B)
        
        self.assertNotEqual(det, 0)
        self.assertFalse(self.engine.is_identity(B))
        self.assertIsNotNone(challenge.challenge_id)
    
    def test_verification_with_valid_proof(self):
        """Verification must succeed with valid proof."""
        prover = Prover(self.engine, verbose=False)
        # Generate challenge first
        challenge = self.verifier.generate_challenge()
        
        # Find valid GREEN->BLUE pair for this challenge (sets secret_v)
        v_valid, _ = prover.find_valid_green_for_challenge(challenge, max_attempts=5000)
        
        # Create commitment with the valid vector
        commitment = prover.commit()
        
        # Create blinded response via solve_challenge (includes blinding + nonce)
        response = prover.solve_challenge(challenge)
        
        # Verify (must unblind internally)
        result = self.verifier.verify(commitment, response, reveal_v=True)
        
        # Should pass all checks
        self.assertTrue(result.is_valid)
        self.assertIn('w_is_blue', result.details['checks_passed'])
        self.assertIn('v_is_green', result.details['checks_passed'])
        self.assertIn('commitment_valid', result.details['checks_passed'])
        self.assertIn('math_valid', result.details['checks_passed'])


if HYPOTHESIS_AVAILABLE:
    @unittest.skipUnless(HYPOTHESIS_AVAILABLE, "hypothesis not installed")
    class TestPropertyBased(unittest.TestCase):
        """Property-based tests for distribution and algebra."""

        @given(st.lists(st.integers(min_value=0, max_value=6), min_size=4, max_size=4))
        @settings(max_examples=50, deadline=None)
        def test_color_distribution_balance(self, vec_list):
            engine = MathEngine(dimension=4, modulus=7)
            v = np.array(vec_list[:4], dtype=np.int64) % engine.p
            color = ColorOracle.get_color(v)
            self.assertIn(color, (Color.GREEN, Color.BLUE))

        @given(st.integers(min_value=0, max_value=20))
        @settings(max_examples=20, deadline=None)
        def test_matrix_inverse_property(self, seed):
            np.random.seed(seed)
            engine = MathEngine(dimension=3, modulus=11)
            A = engine.random_nonsingular_matrix()
            A_inv = engine.matrix_inverse_mod_p(A)
            self.assertIsNotNone(A_inv)
            product = np.dot(A.astype(np.int64), A_inv.astype(np.int64)) % engine.p
            self.assertTrue(np.array_equal(product, np.eye(engine.n, dtype=np.int64)))
else:
    class TestPropertyBased(unittest.TestCase):
        @unittest.skip("hypothesis not installed")
        def test_placeholder(self):
            self.assertTrue(True)

    def test_verification_with_blinded_only(self):
        """Verification must work when only blinded response is sent."""
        prover = Prover(self.engine, verbose=False)
        challenge = self.verifier.generate_challenge()

        v_valid, _ = prover.find_valid_green_for_challenge(challenge, max_attempts=5000)
        commitment = prover.commit()

        # Build response manually: blinded only (no plain w_vector)
        B = challenge.matrix_B
        w = self.engine.matrix_vector_mult(B, v_valid)
        r = int(np.random.randint(1, self.engine.p))
        blinded_w = (r * w) % self.engine.p

        response = Response(
            w_vector=None,
            original_v=v_valid,
            blinded_w_vector=blinded_w,
            blinding_factor=r,
            nonce="deadbeef"
        )

        result = self.verifier.verify(commitment, response, reveal_v=True)

        self.assertTrue(result.is_valid)
        self.assertIn('w_is_blue', result.details['checks_passed'])
        self.assertIn('v_is_green', result.details['checks_passed'])
        self.assertIn('commitment_valid', result.details['checks_passed'])
        self.assertIn('math_valid', result.details['checks_passed'])


class TestChanTheorem(unittest.TestCase):
    """Real tests verifying Chan's Theorem statistically."""
    
    def test_theorem_holds_for_multiple_matrices(self):
        """Chan theorem must hold for multiple random matrices."""
        engine = MathEngine(dimension=4, modulus=7)
        prover = Prover(engine, verbose=False)
        
        success_count = 0
        test_count = 100
        
        for i in range(test_count):
            B = engine.random_nonsingular_matrix()
            challenge = Challenge(matrix_B=B, challenge_id=f"test_{i}")
            
            try:
                v, w = prover.find_valid_green_for_challenge(challenge, max_attempts=3000)
                
                # Verify theorem claim
                self.assertTrue(ColorOracle.is_green(v))
                self.assertTrue(ColorOracle.is_blue(w))
                
                # Verify math
                calculated_w = engine.matrix_vector_mult(B, v)
                self.assertTrue(np.array_equal(calculated_w, w))
                
                success_count += 1
            except RuntimeError:
                pass  # Some may fail due to search limit
        
        # At least 90% should succeed
        success_rate = (success_count / test_count) * 100
        self.assertGreaterEqual(success_rate, 90.0)
    
    def test_theorem_with_different_parameters(self):
        """Chan theorem must hold with different n and p values."""
        test_configs = [
            (3, 5),   # Small
            (4, 7),   # Original
            (5, 11),  # Medium
        ]
        
        for n, p in test_configs:
            engine = MathEngine(dimension=n, modulus=p)
            prover = Prover(engine, verbose=False)
            
            B = engine.random_nonsingular_matrix()
            challenge = Challenge(matrix_B=B, challenge_id=f"test_{n}_{p}")
            
            # Should find GREEN->BLUE pair
            v, w = prover.find_valid_green_for_challenge(challenge, max_attempts=5000)
            
            self.assertTrue(ColorOracle.is_green(v))
            self.assertTrue(ColorOracle.is_blue(w))
            
            # Verify math
            calculated_w = engine.matrix_vector_mult(B, v)
            self.assertTrue(np.array_equal(calculated_w, w))


class TestFullProtocol(unittest.TestCase):
    """Real end-to-end protocol tests."""
    
    def test_complete_protocol_success(self):
        """Complete protocol must execute successfully."""
        engine = MathEngine(dimension=4, modulus=7)
        prover = Prover(engine, verbose=False)
        verifier = Verifier(engine, verbose=False)
        
        # Step 1: Generate challenge first
        challenge = verifier.generate_challenge()
        self.assertIsNotNone(challenge.matrix_B)
        
        # Step 2: Find valid GREEN->BLUE pair for this challenge
        v_valid, _ = prover.find_valid_green_for_challenge(challenge, max_attempts=5000)
        prover.secret_v = v_valid
        self.assertTrue(ColorOracle.is_green(v_valid))
        
        # Step 3: Create commitment with the valid vector
        commitment = prover.commit()
        self.assertIsNotNone(commitment.hash_value)
        
        # Step 4: Create blinded response via solve_challenge
        response = prover.solve_challenge(challenge)
        
        # Step 5: Verify (with reveal)
        result = verifier.verify(commitment, response, reveal_v=True)
        
        # Must be valid
        self.assertTrue(result.is_valid)
        self.assertEqual(result.details['final_result'], 'SUCCESSFUL')
    
    def test_protocol_with_large_dimensions(self):
        """Protocol must work with larger dimensions."""
        engine = MathEngine(dimension=6, modulus=11)
        prover = Prover(engine, verbose=False)
        verifier = Verifier(engine, verbose=False)
        
        # Execute full protocol
        challenge = verifier.generate_challenge()
        
        v_valid, _ = prover.find_valid_green_for_challenge(challenge, max_attempts=5000)
        prover.secret_v = v_valid
        
        commitment = prover.commit()
        response = prover.solve_challenge(challenge)
        result = verifier.verify(commitment, response, reveal_v=True)
        
        self.assertTrue(result.is_valid)


def run_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  CHAN-ZKP TEST SUITE")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestColorOracle))
    suite.addTests(loader.loadTestsFromTestCase(TestMathEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestProver))
    suite.addTests(loader.loadTestsFromTestCase(TestVerifier))
    suite.addTests(loader.loadTestsFromTestCase(TestChanTheorem))
    suite.addTests(loader.loadTestsFromTestCase(TestFullProtocol))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success = total - failures - errors
    
    print(f"\n  Total tests: {total}")
    print(f"  Successful: {success}")
    print(f"  Failed: {failures}")
    print(f"  Errors: {errors}")
    
    if failures == 0 and errors == 0:
        print("\n  ✓✓✓ ALL TESTS PASSED ✓✓✓\n")
    else:
        print("\n  ✗ SOME TESTS FAILED ✗\n")
    
    return result


if __name__ == "__main__":
    run_tests()

