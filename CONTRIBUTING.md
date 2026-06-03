# Contributing to RespAnno

Thank you for your interest in contributing to RespAnno!

## Getting Started

1. Fork the repository and clone it locally.
2. Create a conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate respanno
   ```
3. Install dev dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

1. Create a feature branch from `main`.
2. Make your changes.
3. Ensure all tests pass:
   ```bash
   python -m pytest tests -q
   ```
4. Ensure new code compiles without errors:
   ```bash
   python -m py_compile <changed_file>.py
   ```
5. Commit with a descriptive message and open a pull request.

## Code Style

- Follow PEP 8.
- Write docstrings for all public functions and classes.
- Keep GUI code in the main application file; pure logic belongs in `respanno/`.

## Testing

All new features should include tests. The test suite is organized as:

- `tests/test_*_basic.py` — unit tests for individual modules
- `tests/test_annotation_*.py` — integration tests for annotation workflows
- `tests/test_e2e_ml_pipeline.py` — end-to-end ML pipeline tests

## Pull Request Process

1. Update the CHANGELOG.md with your changes.
2. Ensure the test suite passes.
3. A maintainer will review your PR.

## Questions?

Open an issue on GitHub.
