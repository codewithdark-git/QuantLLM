# Contributing to QuantLLM

We love your input! We want to make contributing to QuantLLM as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`
2. If you've added code that should be tested, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code lints
6. Issue that pull request!

## Development Setup

1. Clone your fork:
```bash
git clone https://github.com/codewithdark-git/QuantLLM.git
cd QuantLLM
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Testing

Run tests with:
```bash
pytest tests/
```

For coverage report:
```bash
pytest --cov=quantllm tests/
```

## Code Style

- We use [black](https://github.com/psf/black) for Python code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [flake8](https://flake8.pycqa.org/) for style guide enforcement

Run linting:
```bash
black .
isort .
flake8 .
```

## Documentation

- Documentation is written in reStructuredText
- Built using Sphinx
- Hosted on Read the Docs

Build docs locally:
```bash
cd docs
make html
```

## Pull Request Process

1. Update the README.md with details of changes to the interface
2. Update the docs/ directory with any new documentation
3. Update the tests/ directory with relevant test cases
4. The PR will be merged once you have the sign-off of two other developers

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/codfewithdark-git/QuantLLM/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/codewithdark-git/QuantLLM/issues/new).

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## License

By contributing, you agree that your contributions will be licensed under its MIT License.