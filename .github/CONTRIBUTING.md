# Contributing to Chan-ZKP

Thank you for your interest in contributing to Chan-ZKP! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/chan-zkp/issues)
2. If not, create a new issue using the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md)
3. Provide as much detail as possible (steps to reproduce, environment, error messages)

### Suggesting Features

1. Check if the feature has already been suggested in [Issues](https://github.com/yourusername/chan-zkp/issues) or [Discussions](https://github.com/yourusername/chan-zkp/discussions)
2. If not, create a new issue using the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md)
3. Explain the motivation and proposed solution

### Asking Questions

- Use [Discussions](https://github.com/yourusername/chan-zkp/discussions) for general questions
- Use [Issues](https://github.com/yourusername/chan-zkp/issues) with the Question template for specific technical questions

## Development Workflow

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/chan-zkp.git
cd chan-zkp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. Make your changes and test them:
   ```bash
   # Run tests
   python tests/test_chan.py
   # or
   pytest tests/test_chan.py -v
   ```

3. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

4. Push to your fork and create a Pull Request

### Co-authored Commits

If you're working with others, you can create co-authored commits:

```bash
git commit -m "Your commit message

Co-authored-by: Name <email@example.com>"
```

This helps with the **Pair Extraordinaire** achievement!

## Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for functions and classes
- Keep functions focused and small

## Testing

- Write tests for new features
- Ensure all tests pass before submitting a PR
- Aim for good test coverage

## Pull Request Process

1. Update README.md if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update documentation if needed
5. Create a clear PR description

## Quickdraw Achievement Tips

Want to earn the **Quickdraw** achievement? Here's how:

1. Look for small, quick fixes (typos, documentation, small bugs)
2. Open an issue for it
3. Fix it within 5 minutes
4. Create a PR and merge it
5. Close the issue

Examples of quick fixes:
- Typo in README
- Missing docstring
- Small bug fix
- Code formatting

## Questions?

Feel free to open a discussion or issue if you have questions about contributing!

