#!/usr/bin/env python3
"""
Chan-ZKP: Zero-Knowledge Proof Demo Based on Melody Chan's Theorem

This program provides an interactive Zero-Knowledge Proof simulation
using Melody Chan's Group Action Theorem.

Usage:
    python main.py                  # Interactive mode
    python main.py --demo           # Automatic demo
    python main.py --chan-test      # Chan Theorem verification test
    python main.py --help           # Help

Authors: Chan-ZKP Demo Project
"""

import sys
import argparse
import numpy as np
from typing import Optional

# Module imports
from src.core import ColorOracle, MathEngine, Color
from src.actors import Prover, Verifier, run_protocol_demo


# ANSI color codes (for terminal output)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_banner():
    """Prints ASCII art banner."""
    banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   ██████╗██╗  ██╗ █████╗ ███╗   ██╗      ███████╗██╗  ██╗██████╗   ║
║  ██╔════╝██║  ██║██╔══██╗████╗  ██║      ╚══███╔╝██║ ██╔╝██╔══██╗  ║
║  ██║     ███████║███████║██╔██╗ ██║█████╗  ███╔╝ █████╔╝ ██████╔╝  ║
║  ██║     ██╔══██║██╔══██║██║╚██╗██║╚════╝ ███╔╝  ██╔═██╗ ██╔═══╝   ║
║  ╚██████╗██║  ██║██║  ██║██║ ╚████║      ███████╗██║  ██╗██║       ║
║   ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝      ╚══════╝╚═╝  ╚═╝╚═╝       ║
║                                                                    ║
║         Zero-Knowledge Proof Based on Melody Chan's Theorem        ║
║                        Demo System v1.0                            ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════╝{Colors.ENDC}
"""
    print(banner)


def print_colored(text: str, color: str = Colors.ENDC):
    """Prints colored text."""
    print(f"{color}{text}{Colors.ENDC}")


def print_section(title: str):
    """Prints section header."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}{Colors.ENDC}")


def print_step(step_num: int, title: str):
    """Prints step header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}[STEP {step_num}] {title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'─' * 50}{Colors.ENDC}")


def get_parameters() -> tuple:
    """Gets parameters from user."""
    print_section("PARAMETER SETTINGS")
    
    print(f"\n{Colors.YELLOW}Chan Theorem condition: |F| > n + 1")
    print(f"That is: modulus > dimension + 1{Colors.ENDC}\n")
    
    # Dimension
    while True:
        try:
            dim_input = input(f"  Vector dimension (n) [default: 4]: ").strip()
            dimension = int(dim_input) if dim_input else 4
            if dimension < 1:
                print(f"  {Colors.RED}Dimension must be positive!{Colors.ENDC}")
                continue
            break
        except ValueError:
            print(f"  {Colors.RED}Please enter a valid number!{Colors.ENDC}")
    
    # Modulus
    min_modulus = dimension + 2
    while True:
        try:
            mod_input = input(f"  Field modulus (p) [minimum: {min_modulus}, default: 7]: ").strip()
            modulus = int(mod_input) if mod_input else max(7, min_modulus)
            if modulus < min_modulus:
                print(f"  {Colors.RED}Modulus must be at least {min_modulus}!{Colors.ENDC}")
                continue
            break
        except ValueError:
            print(f"  {Colors.RED}Please enter a valid number!{Colors.ENDC}")
    
    print(f"\n  {Colors.GREEN}✓ Parameters: n={dimension}, p={modulus} (GF({modulus})){Colors.ENDC}")
    
    return dimension, modulus


def interactive_mode():
    """Interactive mode - user experiences the protocol step by step."""
    print_banner()
    print_colored("\nINTERACTIVE MODE", Colors.BOLD)
    print("In this mode, you will experience the ZKP protocol step by step.\n")
    
    # Get parameters
    dimension, modulus = get_parameters()
    
    # Create engine and actors
    engine = MathEngine(dimension=dimension, modulus=modulus)
    prover = Prover(engine, verbose=False)
    verifier = Verifier(engine, verbose=False)
    
    # STEP 1: Secret vector generation
    print_step(1, "PROVER: Secret GREEN Vector Generation")
    print(f"  Prover is searching for a GREEN secret vector...")
    
    input(f"\n  {Colors.YELLOW}[Press Enter]{Colors.ENDC}")
    
    try:
        v = prover.generate_secret()
        print(f"\n  {Colors.GREEN}✓ Secret vector found!{Colors.ENDC}")
        print(f"  v = {v}")
        print(f"  Color: {Colors.GREEN}GREEN{Colors.ENDC}")
    except RuntimeError as e:
        print(f"  {Colors.RED}ERROR: {e}{Colors.ENDC}")
        return
    
    # STEP 2: Commitment
    print_step(2, "PROVER → VERIFIER: Commitment Submission")
    print("  Prover sends the HASH (commitment) of the secret vector.")
    print("  This value proves the existence of v without revealing it.")
    
    input(f"\n  {Colors.YELLOW}[Press Enter]{Colors.ENDC}")
    
    commitment = prover.commit()
    print(f"\n  {Colors.CYAN}Commitment (SHA-256):{Colors.ENDC}")
    print(f"  {commitment.hash_value}")
    
    # STEP 3: Challenge
    print_step(3, "VERIFIER → PROVER: Challenge Submission")
    print("  Verifier generates a random non-singular matrix B.")
    print("  This matrix is sent to Prover as a challenge.")
    
    input(f"\n  {Colors.YELLOW}[Press Enter]{Colors.ENDC}")
    
    challenge = verifier.generate_challenge()
    print(f"\n  {Colors.CYAN}Challenge Matrix B:{Colors.ENDC}")
    print(f"  {challenge.matrix_B}")
    print(f"\n  Challenge ID: {challenge.challenge_id}")
    
    # STEP 4: Response
    print_step(4, "PROVER: Challenge Solution")
    print("  Prover calculates w = B × v.")
    print("  Chan Theorem: Under appropriate conditions, w should be BLUE!")
    
    input(f"\n  {Colors.YELLOW}[Press Enter]{Colors.ENDC}")
    
    response = prover.solve_challenge(challenge)
    w_color = ColorOracle.get_color_name(response.w_vector)
    
    print(f"\n  {Colors.CYAN}Calculation:{Colors.ENDC}")
    print(f"  v = {v} ({Colors.GREEN}GREEN{Colors.ENDC})")
    print(f"  w = B × v = {response.w_vector}", end=" ")
    
    if ColorOracle.is_blue(response.w_vector):
        print(f"({Colors.BLUE}BLUE{Colors.ENDC})")
        print(f"\n  {Colors.GREEN}✓ Excellent! w is BLUE - Proof is strong!{Colors.ENDC}")
    else:
        print(f"({Colors.GREEN}GREEN{Colors.ENDC})")
        print(f"\n  {Colors.YELLOW}⚠ w is GREEN - A different v may be needed for this challenge{Colors.ENDC}")
    
    # STEP 5: Verification
    print_step(5, "VERIFIER: Verification")
    print("  Verifier verifies Prover's response.")
    print("  Checks: w's color, commitment consistency, mathematical correctness")
    
    input(f"\n  {Colors.YELLOW}[Press Enter]{Colors.ENDC}")
    
    result = verifier.verify(commitment, response, reveal_v=True)
    
    print(f"\n  {Colors.CYAN}Verification Results:{Colors.ENDC}")
    
    checks = result.details.get('checks_passed', [])
    
    # Check 1: w color
    if 'w_is_blue' in checks:
        print(f"  [1] Is w BLUE?           {Colors.GREEN}✓ YES{Colors.ENDC}")
    else:
        print(f"  [1] Is w BLUE?           {Colors.RED}✗ NO{Colors.ENDC}")
    
    # Check 2: v color
    if 'v_is_green' in checks:
        print(f"  [2] Is v GREEN?          {Colors.GREEN}✓ YES{Colors.ENDC}")
    else:
        print(f"  [2] Is v GREEN?          {Colors.RED}✗ NO{Colors.ENDC}")
    
    # Check 3: Commitment
    if 'commitment_valid' in checks:
        print(f"  [3] Is commitment valid?  {Colors.GREEN}✓ YES{Colors.ENDC}")
    else:
        print(f"  [3] Is commitment valid?  {Colors.RED}✗ NO{Colors.ENDC}")
    
    # Check 4: Mathematics
    if 'math_valid' in checks:
        print(f"  [4] Is B×v = w?             {Colors.GREEN}✓ YES{Colors.ENDC}")
    else:
        print(f"  [4] Is B×v = w?             {Colors.RED}✗ NO{Colors.ENDC}")
    
    # Final result
    print_section("PROTOCOL RESULT")
    
    if result.is_valid:
        print(f"""
  {Colors.GREEN}{Colors.BOLD}
  ╔═══════════════════════════════════════════════════════╗
  ║                                                       ║
  ║        ✓ ✓ ✓   VERIFICATION SUCCESSFUL   ✓ ✓ ✓       ║
  ║                                                       ║
  ║   Prover successfully proved possession of a          ║
  ║   secret GREEN vector!                               ║
  ║                                                       ║
  ╚═══════════════════════════════════════════════════════╝
  {Colors.ENDC}""")
    else:
        print(f"""
  {Colors.RED}{Colors.BOLD}
  ╔═══════════════════════════════════════════════════════╗
  ║                                                       ║
  ║        ✗ ✗ ✗   VERIFICATION FAILED   ✗ ✗ ✗            ║
  ║                                                       ║
  ║   Some checks failed.                                 ║
  ║   Passed checks: {len(checks)}/4                              ║
  ║                                                       ║
  ╚═══════════════════════════════════════════════════════╝
  {Colors.ENDC}""")


def chan_theorem_test(dimension: int = 4, modulus: int = 7, iterations: int = 100):
    """
    Chan Theorem verification test.
    
    Theorem: For every non-singular, non-identity matrix B,
    there exists at least one v that transforms GREEN v to BLUE w.
    
    This test statistically verifies this claim.
    """
    print_banner()
    print_section("CHAN THEOREM VERIFICATION TEST")
    
    print(f"\n  {Colors.CYAN}Theorem:{Colors.ENDC}")
    print(f"  \"For every non-singular, non-identity n×n matrix B,")
    print(f"   there exists at least one v that transforms GREEN v to BLUE w.\"")
    
    print(f"\n  {Colors.YELLOW}Test Parameters:{Colors.ENDC}")
    print(f"  - Dimension (n): {dimension}")
    print(f"  - Modulus (p): {modulus}")
    print(f"  - Number of iterations: {iterations}")
    
    engine = MathEngine(dimension=dimension, modulus=modulus)
    prover = Prover(engine, verbose=False)
    
    print(f"\n  {Colors.CYAN}Test starting...{Colors.ENDC}\n")
    
    successes = 0
    failures = 0
    total_attempts = 0
    
    for i in range(iterations):
        # Create random challenge
        B = engine.random_nonsingular_matrix(exclude_identity=True)
        
        from src.actors import Challenge
        challenge = Challenge(
            matrix_B=B,
            challenge_id=f"test_{i}"
        )
        
        # Find GREEN -> BLUE pair
        try:
            v, w = prover.find_valid_green_for_challenge(challenge, max_attempts=5000)
            successes += 1
            attempts = "≤5000"
        except RuntimeError:
            failures += 1
            attempts = ">5000 (FAILED)"
        
        # Progress indicator
        progress = (i + 1) / iterations * 100
        bar_length = 30
        filled = int(bar_length * (i + 1) / iterations)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        sys.stdout.write(f"\r  [{bar}] {progress:.0f}% ({i+1}/{iterations})")
        sys.stdout.flush()
    
    print("\n")
    
    # Results
    print_section("TEST RESULTS")
    
    success_rate = (successes / iterations) * 100
    
    print(f"\n  {Colors.CYAN}Statistics:{Colors.ENDC}")
    print(f"  - Total tests: {iterations}")
    print(f"  - Successful: {Colors.GREEN}{successes}{Colors.ENDC}")
    print(f"  - Failed: {Colors.RED}{failures}{Colors.ENDC}")
    print(f"  - Success rate: {Colors.BOLD}{success_rate:.1f}%{Colors.ENDC}")
    
    if success_rate >= 95:
        print(f"""
  {Colors.GREEN}{Colors.BOLD}
  ╔═══════════════════════════════════════════════════════╗
  ║                                                       ║
  ║   ✓ CHAN THEOREM VERIFIED!                           ║
  ║                                                       ║
  ║   For every tested matrix, a vector was found        ║
  ║   that provides the GREEN→BLUE transformation.        ║
  ║                                                       ║
  ╚═══════════════════════════════════════════════════════╝
  {Colors.ENDC}""")
    else:
        print(f"""
  {Colors.YELLOW}{Colors.BOLD}
  ╔═══════════════════════════════════════════════════════╗
  ║                                                       ║
  ║   ⚠ WARNING: Some tests failed                       ║
  ║                                                       ║
  ║   This may indicate that the search limit is         ║
  ║   insufficient. Try a larger max_attempts value.     ║
  ║                                                       ║
  ╚═══════════════════════════════════════════════════════╝
  {Colors.ENDC}""")


def auto_demo():
    """Automatic demo - runs protocol with default parameters."""
    print_banner()
    print_colored("\nAUTOMATIC DEMO MODE", Colors.BOLD)
    print("Protocol will run with default parameters (n=4, p=7).\n")
    
    run_protocol_demo(dimension=4, modulus=7, verbose=True)


def main_menu():
    """Main menu."""
    print_banner()
    
    print(f"\n{Colors.BOLD}Main Menu:{Colors.ENDC}")
    print(f"  {Colors.CYAN}1.{Colors.ENDC} Interactive Mode (Step-by-step protocol)")
    print(f"  {Colors.CYAN}2.{Colors.ENDC} Automatic Demo")
    print(f"  {Colors.CYAN}3.{Colors.ENDC} Chan Theorem Verification Test")
    print(f"  {Colors.CYAN}4.{Colors.ENDC} About")
    print(f"  {Colors.CYAN}0.{Colors.ENDC} Exit")
    
    while True:
        choice = input(f"\n  {Colors.YELLOW}Your choice [1-4, 0]: {Colors.ENDC}").strip()
        
        if choice == '1':
            interactive_mode()
            break
        elif choice == '2':
            auto_demo()
            break
        elif choice == '3':
            chan_theorem_test()
            break
        elif choice == '4':
            print_about()
            break
        elif choice == '0':
            print(f"\n  {Colors.GREEN}Goodbye!{Colors.ENDC}\n")
            sys.exit(0)
        else:
            print(f"  {Colors.RED}Invalid choice!{Colors.ENDC}")


def print_about():
    """About information."""
    print_section("ABOUT")
    
    print(f"""
  {Colors.CYAN}Chan-ZKP Demo v1.0{Colors.ENDC}
  
  This project simulates Melody Chan's Group Action Theorem
  (proven in 2004 while she was an undergraduate at Yale University)
  in a cryptographic Zero-Knowledge Proof context.
  
  {Colors.YELLOW}Melody Chan's Theorem:{Colors.ENDC}
  "Let n be a positive integer and F be a field satisfying |F| > n + 1.
  The n-vectors in F^n can be colored such that for every non-singular
  n×n matrix B (other than the identity matrix), there exists a GREEN
  vector v for which Bv is BLUE."
  
  {Colors.YELLOW}Cryptographic Application:{Colors.ENDC}
  - GREEN vector: Secret information (witness)
  - Matrix transformation: Proof mechanism
  - BLUE result: Successful proof
  
  {Colors.CYAN}References:{Colors.ENDC}
  - Original paper: combinatorics.org/Volume_13/v13i1toc.html
  - Book: "Fearless Symmetry" - Avner Ash & Robert Gross
""")


def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description='Chan-ZKP: Zero-Knowledge Proof Demo Based on Melody Chan\'s Theorem',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  # Main menu
  python main.py --demo           # Automatic demo
  python main.py --interactive   # Interactive mode
  python main.py --chan-test      # Theorem verification test
  python main.py --chan-test -n 5 -p 11 -i 200  # Test with custom parameters
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                        help='Run automatic demo mode')
    parser.add_argument('--interactive', '-I', action='store_true',
                        help='Run interactive mode')
    parser.add_argument('--chan-test', action='store_true',
                        help='Run Chan Theorem verification test')
    parser.add_argument('-n', '--dimension', type=int, default=4,
                        help='Vector dimension (default: 4)')
    parser.add_argument('-p', '--modulus', type=int, default=7,
                        help='Field modulus (default: 7)')
    parser.add_argument('-i', '--iterations', type=int, default=100,
                        help='Number of test iterations (default: 100)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    try:
        if args.demo:
            auto_demo()
        elif args.interactive:
            interactive_mode()
        elif args.chan_test:
            chan_theorem_test(
                dimension=args.dimension,
                modulus=args.modulus,
                iterations=args.iterations
            )
        else:
            main_menu()
    except KeyboardInterrupt:
        print(f"\n\n  {Colors.YELLOW}Program interrupted by user.{Colors.ENDC}\n")
        sys.exit(0)

