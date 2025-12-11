# GitHub Actions Workflows

## Available Workflows

### `test.yml` (Full)
- Tests on multiple OS (Ubuntu, macOS)
- Tests on Python 3.8-3.12
- Includes type checking and linting
- Includes benchmark job
- **Uses more GitHub Actions minutes**

### `test-light.yml` (Lightweight)
- Tests only on Ubuntu
- Tests on Python 3.10, 3.11, 3.12 (most common versions)
- Basic test suite only
- **Uses fewer GitHub Actions minutes**

## Billing Issue?

If you see "account is locked due to a billing issue":

1. **For Public Repositories**: GitHub Actions are free with unlimited minutes
2. **For Private Repositories**: 
   - Free tier: 2,000 minutes/month
   - You may need to add a payment method or upgrade

### Quick Fix Options:

1. **Use the lightweight workflow**: Rename `test-light.yml` to `test.yml`
2. **Test locally**: Run `python tests/test_chan.py` or `pytest tests/test_chan.py`
3. **Fix billing**: Go to GitHub Settings → Billing → Actions

## Local Testing

You can test everything locally:

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
python tests/test_chan.py

# Run with pytest and coverage
pytest tests/test_chan.py -v --cov=src --cov-report=term-missing

# Run type checking
mypy src/ --ignore-missing-imports

# Run linting
flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
```

