# Chan-ZKP: Building Zero-Knowledge Proofs from Group Actions

> **Zero-knowledge proof system based on Melody Chan's theorem using linear algebra and matrix operations. Post-quantum friendly, no trusted setup required.**

Ever wondered if you could prove you know a secret without revealing it? That's what Zero-Knowledge Proofs (ZKPs) are all about. This project implements a ZKP system based on a cool theorem from 2004 by Melody Chan.

## What's the Big Idea?

Most ZKP systems you've heard of (like zk-SNARKs) rely on elliptic curves or polynomial commitments. This one is different - it's built on **linear algebra and combinatorics**. Specifically, it uses Chan's theorem about coloring vectors in finite fields.

### The Math Behind It

Here's what Chan proved: Take a finite field F (like GF(7) or GF(11)) where the field size is bigger than your vector dimension plus one. You can color all the n-vectors in F^n with two colors (let's call them GREEN and BLUE) such that:

**For every invertible n×n matrix B (except the identity), there's always at least one GREEN vector v that transforms into a BLUE vector when you multiply it by B.**

In other words: `B × v = w`, where v is GREEN and w is BLUE.

Why is this useful for crypto? Because it gives us a **binding property** - you can commit to a GREEN vector, and when challenged with a matrix B, you can prove you know a GREEN vector that maps to BLUE under that transformation. The verifier never sees your original vector, but they can verify the math checks out.

### How We Use It

- **GREEN vectors** = your secret (the witness)
- **Matrix B** = the challenge from the verifier
- **BLUE result** = proof that you knew a GREEN vector without revealing it

The cool part? This is all just matrix multiplication mod p. No fancy elliptic curves, no complex polynomial math. Just good old linear algebra.

## Getting Started

You'll need Python 3.7+ and numpy. That's it.

```bash
# Clone and setup
cd chan_zkp
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Then just run:

```bash
python main.py
```

You'll get a menu with a few options. The interactive mode is probably the most fun - it walks you through the protocol step by step so you can see what's happening.

## Command Line Usage

The CLI is pretty straightforward:

```bash
# Main menu (interactive)
python main.py

# Quick demo with defaults
python main.py --demo

# Interactive walkthrough
python main.py --interactive

# Test the theorem with custom params
python main.py --chan-test -n 5 -p 11 -i 200
```

Full argument list:

| Argument        | Short | What it does                  | Default |
| --------------- | ----- | ----------------------------- | ------- |
| `--demo`        | -     | Run automatic demo            | -       |
| `--interactive` | `-I`  | Interactive mode              | -       |
| `--chan-test`   | -     | Test Chan's theorem           | -       |
| `--dimension`   | `-n`  | Vector size (n)               | 4       |
| `--modulus`     | `-p`  | Field size (p, must be prime) | 7       |
| `--iterations`  | `-i`  | How many tests to run         | 100     |
| `--help`        | `-h`  | Show help                     | -       |

**Important:** The theorem requires `p > n + 1`. If you try something like n=4, p=5, it'll throw an error because 5 is not greater than 4+1.

## How the Protocol Works

The protocol follows a standard ZKP pattern:

1. **Prover** generates a secret GREEN vector v (this is the "mining" step - keep trying random vectors until you get a GREEN one)

2. **Prover** sends a commitment: `SHA-256(v)`. This is a hash, so the verifier can't reverse it to get v.

3. **Verifier** sends a challenge: a random invertible matrix B. This is the "prove it" moment.

4. **Prover** calculates `w = B × v` and sends w back. If everything works out, w should be BLUE.

5. **Verifier** checks:
   - Is w actually BLUE? (Chan's theorem says it should be)
   - If in reveal mode: Is v GREEN? Does the hash match? Does the math work?

The beauty is that the verifier learns nothing about v except that you knew a GREEN vector. That's the "zero-knowledge" part.

## Security Notes & Blinding

- **Response blinding:** The prover multiplies `w` by a random non-zero scalar `r` (mod p) and sends `r·w` plus `r`. The verifier unblinds using `r^{-1}` mod p. This prevents raw `w` from being sent directly on the wire.
- **What it protects:** Simple scalar blinding hides the raw response vector from passive observers. It also prevents deterministic replays of `w`.
- **What it does NOT protect:** A malicious verifier still learns `w` after unblinding (they know `r`). Direction of `w` is not hidden—only scaled. There is no transcript privacy or side-channel hardening. This is a PoC, not a production ZK protocol.
- **Threat model:** Honest-but-curious verifier, no network adversary altering messages, no timing/power side-channel protections.
- **Parameters:** The theorem requires prime `p > n + 1`. Current defaults use small demo primes (e.g., p=7). For real security you'd need much larger primes/fields, stronger hashing, formal proofs, and hardened blinding.
- **Tests:** Blinded verification is covered in the test suite (19 real tests, no mocks).

## Running Tests

```bash
python tests/test_chan.py
```

The tests cover:

- Color oracle determinism and distribution
- Matrix operations (inverse, multiplication, determinant)
- Prover/Verifier protocol correctness
- Chan's theorem verification across different parameters
- Full end-to-end protocol runs

All tests should pass. If they don't, something's broken.

The code is modular - `core.py` handles all the math (matrix operations, color functions), and `actors.py` implements the protocol logic. You can easily extend it or use the components separately.

## Why This Matters

Most ZKP systems are either:

- Based on elliptic curves (vulnerable to quantum computers)
- Require trusted setups
- Need complex polynomial math

This approach is different. It's:

- **Post-quantum friendly** (matrix problems are hard for quantum computers too)
- **No trusted setup** needed
- **Simple math** (just linear algebra)
- **Lightweight** (matrix multiplication is fast, even on small devices)

Is it production-ready? Not yet. This is a proof-of-concept. For real use, you'd need:

- Better blinding techniques
- Stronger hash functions
- Formal security proofs
- Performance tuning

But it's a solid foundation for exploring matrix-based ZKPs.

## Project Structure

```
chan_zkp/
├── src/
│   ├── core.py       # Math engine and color oracle
│   └── actors.py     # Prover and Verifier classes
├── tests/
│   └── test_chan.py  # 18 real tests (no mocks)
├── main.py           # CLI interface
├── requirements.txt  # Dependencies
└── README.md
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

Educational/research use. Feel free to fork, modify, and experiment.

---

Built to explore what's possible when you combine combinatorics, linear algebra, and cryptography. The theorem is from Melody Chan (Yale, 2004). The implementation is ours.

## GitHub Repository Setup

**Repository Description** (for GitHub settings):

```
Matrix-based zero-knowledge proof system using Melody Chan's theorem. Post-quantum friendly, no trusted setup. Python implementation.
```

**Suggested Topics:** `zero-knowledge-proofs`, `zkp`, `cryptography`, `post-quantum-cryptography`, `matrix-cryptography`, `python`, `linear-algebra`

See [GITHUB_SETUP.md](GITHUB_SETUP.md) for complete GitHub setup instructions.
