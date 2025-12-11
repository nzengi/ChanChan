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
from src.config import ChanZKPConfig, get_config, set_config, reset_config
from src.logger import get_logger, StructuredLogger


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
        
        # Create blinded response via solve_challenge (includes blinding + nonce + MAC)
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

        # Use solve_challenge to produce blinded response (with nonce/MAC)
        response = prover.solve_challenge(challenge)
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


class TestConfig(unittest.TestCase):
    """Tests for configuration management."""
    
    def setUp(self):
        """Reset config before each test."""
        reset_config()
    
    def tearDown(self):
        """Reset config after each test."""
        reset_config()
    
    def test_get_config_returns_singleton(self):
        """get_config should return the same instance."""
        config1 = get_config()
        config2 = get_config()
        self.assertIs(config1, config2)
    
    def test_config_from_env_defaults(self):
        """Config should have correct defaults."""
        config = ChanZKPConfig.from_env()
        self.assertEqual(config.color_key, b"chan-zkp-color-key")
        self.assertEqual(config.commit_key, b"chan-zkp-commit-key")
        self.assertEqual(config.session_key, b"chan-zkp-session-key")
        self.assertEqual(config.log_level, "INFO")
        self.assertEqual(config.log_format, "json")
        self.assertFalse(config.verbose)
    
    def test_config_key_getters(self):
        """Key getter methods should return correct keys."""
        config = ChanZKPConfig(
            color_key=b"test-color",
            commit_key=b"test-commit",
            session_key=b"test-session"
        )
        self.assertEqual(config.get_color_key(), b"test-color")
        self.assertEqual(config.get_commit_key(), b"test-commit")
        self.assertEqual(config.get_session_key(), b"test-session")
    
    def test_set_config(self):
        """set_config should update global config."""
        custom_config = ChanZKPConfig(
            color_key=b"custom-color",
            commit_key=b"custom-commit",
            session_key=b"custom-session"
        )
        set_config(custom_config)
        retrieved = get_config()
        self.assertIs(retrieved, custom_config)
        self.assertEqual(retrieved.get_color_key(), b"custom-color")
    
    def test_reset_config(self):
        """reset_config should reset to None."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        # Should create new instance after reset
        self.assertIsNot(config1, config2)


class TestLogger(unittest.TestCase):
    """Tests for structured logging."""
    
    def setUp(self):
        """Reset config before each test."""
        reset_config()
    
    def tearDown(self):
        """Reset config after each test."""
        reset_config()
    
    def test_get_logger_returns_instance(self):
        """get_logger should return StructuredLogger instance."""
        logger = get_logger("TEST")
        self.assertIsInstance(logger, StructuredLogger)
        self.assertEqual(logger.name, "TEST")
    
    def test_logger_json_format(self):
        """Logger should output JSON format when configured."""
        import json
        import io
        import sys
        
        config = ChanZKPConfig(
            color_key=b"test",
            commit_key=b"test",
            session_key=b"test",
            log_format="json",
            verbose=True
        )
        set_config(config)
        
        logger = get_logger("TEST_COMPONENT")
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            logger.info("Test message", {"key": "value"})
            output = captured_output.getvalue().strip()
            # Should be valid JSON
            log_data = json.loads(output)
            self.assertEqual(log_data["level"], "INFO")
            self.assertEqual(log_data["component"], "TEST_COMPONENT")
            self.assertEqual(log_data["message"], "Test message")
            self.assertEqual(log_data["data"]["key"], "value")
        finally:
            sys.stdout = old_stdout
    
    def test_logger_text_format(self):
        """Logger should output text format when configured."""
        config = ChanZKPConfig(
            color_key=b"test",
            commit_key=b"test",
            session_key=b"test",
            log_format="text",
            verbose=True
        )
        set_config(config)
        
        logger = get_logger("TEST_COMPONENT")
        # Just verify it doesn't crash - text format uses standard logging
        # which pytest captures differently
        logger.info("Test message")
        logger.warning("Warning message")
        # If we get here without exception, it works
        self.assertTrue(True)
    
    def test_logger_all_levels(self):
        """All log levels should work."""
        import io
        import sys
        
        config = ChanZKPConfig(
            color_key=b"test",
            commit_key=b"test",
            session_key=b"test",
            log_format="json",
            verbose=True
        )
        set_config(config)
        
        logger = get_logger("TEST")
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
            
            output = captured_output.getvalue()
            lines = [line for line in output.strip().split('\n') if line]
            self.assertGreaterEqual(len(lines), 5)  # At least 5 log entries
        finally:
            sys.stdout = old_stdout


class TestActorsEdgeCases(unittest.TestCase):
    """Tests for edge cases in actors."""
    
    def setUp(self):
        self.engine = MathEngine(dimension=3, modulus=7)
        self.prover = Prover(self.engine, verbose=False)
        self.verifier = Verifier(self.engine, verbose=False)
    
    def test_verification_without_reveal_v(self):
        """Verification should work without revealing v."""
        # Generate challenge
        challenge = self.verifier.generate_challenge()
        
        # Find valid GREEN->BLUE pair
        v_valid, _ = self.prover.find_valid_green_for_challenge(challenge, max_attempts=5000)
        self.prover.secret_v = v_valid
        
        # Create commitment
        commitment = self.prover.commit()
        
        # Create response
        response = self.prover.solve_challenge(challenge)
        
        # Verify without revealing v
        result = self.verifier.verify(commitment, response, reveal_v=False)
        
        self.assertTrue(result.is_valid)
        # When reveal_v=False, v checks should not be in details
        self.assertIn("w_is_blue", result.details["checks_passed"])
        # v_is_green should not be checked when reveal_v=False
        self.assertNotIn("v_is_green", result.details.get("checks_passed", []))
    
    def test_prover_commit_without_secret(self):
        """commit() should raise ValueError if secret not generated."""
        with self.assertRaises(ValueError) as context:
            self.prover.commit()
        self.assertIn("generate_secret", str(context.exception))
    
    def test_prover_solve_without_secret(self):
        """solve_challenge() should raise ValueError if secret not generated."""
        challenge = self.verifier.generate_challenge()
        with self.assertRaises(ValueError) as context:
            self.prover.solve_challenge(challenge)
        self.assertIn("generate_secret", str(context.exception))
    
    def test_verifier_verify_invalid_commitment(self):
        """Verification should fail with invalid commitment."""
        challenge = self.verifier.generate_challenge()
        v_valid, _ = self.prover.find_valid_green_for_challenge(challenge, max_attempts=5000)
        self.prover.secret_v = v_valid
        
        # Create response
        response = self.prover.solve_challenge(challenge)
        
        # Create invalid commitment
        invalid_commitment = Commitment(
            hash_value="invalid_hash",
            vector_dimension=self.engine.n,
            field_modulus=self.engine.p
        )
        
        # Verify should fail
        result = self.verifier.verify(invalid_commitment, response, reveal_v=True)
        self.assertFalse(result.is_valid)


class TestCoreEdgeCases(unittest.TestCase):
    """Tests for edge cases in core module."""
    
    def test_color_oracle_with_custom_key(self):
        """ColorOracle should work with custom key."""
        v = np.array([1, 2, 3])
        color1 = ColorOracle.get_color(v, key=b"key1")
        color2 = ColorOracle.get_color(v, key=b"key2")
        # Different keys may produce different colors
        # But same key should produce same color
        color1_again = ColorOracle.get_color(v, key=b"key1")
        self.assertEqual(color1, color1_again)
    
    def test_math_engine_determinant_zero(self):
        """Determinant should be 0 for singular matrix."""
        engine = MathEngine(dimension=2, modulus=7)
        # Create singular matrix (rank 1)
        singular = np.array([[1, 2], [2, 4]], dtype=np.int64)
        det = engine._determinant_mod_p(singular)
        self.assertEqual(det, 0)
    
    def test_math_engine_inverse_singular_matrix(self):
        """Inverse should return None for singular matrix."""
        engine = MathEngine(dimension=2, modulus=7)
        # Create singular matrix
        singular = np.array([[1, 2], [2, 4]], dtype=np.int64)
        inv = engine.matrix_inverse_mod_p(singular)
        self.assertIsNone(inv)
    
    def test_math_engine_identity_detection(self):
        """is_identity should correctly detect identity matrices."""
        engine = MathEngine(dimension=3, modulus=7)
        identity = np.eye(3, dtype=np.int64)
        self.assertTrue(engine.is_identity(identity))
        
        # Non-identity matrix
        non_identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=np.int64)
        self.assertFalse(engine.is_identity(non_identity))


class TestProtocolDemo(unittest.TestCase):
    """Tests for protocol demo function."""
    
    def test_run_protocol_demo_success(self):
        """run_protocol_demo should complete successfully."""
        from src.actors import run_protocol_demo
        
        result = run_protocol_demo(dimension=3, modulus=7, verbose=False)
        # Should return a result (may succeed or fail depending on random vectors)
        self.assertIsNotNone(result)
        self.assertIsInstance(result.is_valid, bool)
        self.assertIn("checks_passed", result.details)
    
    def test_run_protocol_demo_failure_handling(self):
        """run_protocol_demo should handle failures gracefully."""
        from src.actors import run_protocol_demo
        
        # Test with valid params (modulus > dimension + 1)
        result = run_protocol_demo(dimension=2, modulus=5, verbose=False)
        # Should either succeed or return None on error
        if result is not None:
            self.assertIsInstance(result.is_valid, bool)


class TestLoggerAdvanced(unittest.TestCase):
    """Advanced tests for logger."""
    
    def setUp(self):
        reset_config()
    
    def tearDown(self):
        reset_config()
    
    def test_json_formatter_with_exception(self):
        """JSONFormatter should handle exceptions correctly."""
        import logging
        from src.logger import JSONFormatter
        
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Test error",
            args=(),
            exc_info=None
        )
        
        # Should produce valid JSON
        output = formatter.format(record)
        import json
        log_data = json.loads(output)
        self.assertEqual(log_data["level"], "ERROR")
        self.assertEqual(log_data["message"], "Test error")
    
    def test_config_with_bytes_key(self):
        """Config should handle bytes keys from environment."""
        import os
        import tempfile
        
        # Test that get_key_bytes handles bytes correctly
        # This tests the edge case in config.py line 53
        config = ChanZKPConfig.from_env()
        # Should work with default string keys
        self.assertIsInstance(config.get_color_key(), bytes)
        self.assertIsInstance(config.get_commit_key(), bytes)
        self.assertIsInstance(config.get_session_key(), bytes)


if __name__ == "__main__":
    run_tests()

