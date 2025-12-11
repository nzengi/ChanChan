"""
Chan-ZKP: Zero-Knowledge Proof Demo System Based on Melody Chan's Theorem

This package simulates Melody Chan's Group Action Theorem
in a cryptographic ZKP (Zero-Knowledge Proof) context.

Modules:
    - core: Mathematical core (ColorOracle, MathEngine)
    - actors: Protocol actors (Prover, Verifier)
"""

__version__ = "1.0.0"
__author__ = "Chan-ZKP Demo"

# Public API exports
try:
    from .src.core import ColorOracle, MathEngine, Color
    from .src.actors import Prover, Verifier, Challenge, Commitment, Response, VerificationResult
    
    __all__ = [
        'ColorOracle',
        'MathEngine', 
        'Color',
        'Prover',
        'Verifier',
        'Challenge',
        'Commitment',
        'Response',
        'VerificationResult',
    ]
except ImportError:
    # Fallback for direct src imports (development mode)
    try:
        from src.core import ColorOracle, MathEngine, Color
        from src.actors import Prover, Verifier, Challenge, Commitment, Response, VerificationResult
        
        __all__ = [
            'ColorOracle',
            'MathEngine', 
            'Color',
            'Prover',
            'Verifier',
            'Challenge',
            'Commitment',
            'Response',
            'VerificationResult',
        ]
    except ImportError:
        __all__ = []

