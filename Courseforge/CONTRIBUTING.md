# Contributing to Courseforge

Thank you for your interest in contributing to Courseforge! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) DART installation for document accessibility conversion

### Setup

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Courseforge.git
   cd Courseforge
   ```
3. Install dependencies:
   ```bash
   pip install -r scripts/requirements.txt
   ```
4. Set environment variables (optional):
   ```bash
   export COURSEFORGE_PATH=/path/to/Courseforge
   export DART_PATH=/path/to/DART  # If using DART integration
   ```

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Making Changes

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Test your changes thoroughly
4. Commit with clear, descriptive messages

### Commit Messages

Follow conventional commit format:
- `feat: add new assessment template generator`
- `fix: resolve manifest namespace issue`
- `docs: update agent documentation`
- `refactor: simplify IMSCC packaging logic`

## Code Style

### Python

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions
- Include type hints where appropriate

### Documentation

- Update README files when changing functionality
- Document new features in the appropriate `docs/` file
- Keep CLAUDE.md updated for agent behavior changes

## Pull Request Process

1. Ensure your code follows the project style guidelines
2. Update documentation as needed
3. Test your changes with sample course content
4. Create a pull request with:
   - Clear title describing the change
   - Description of what changed and why
   - Any related issue numbers

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass (if applicable)
- [ ] Documentation updated
- [ ] No hardcoded paths or credentials
- [ ] IMSCC output validates correctly (for packager changes)

## Agent Development

When modifying or creating agents:

1. Follow the agent specification format in `agents/`
2. Update CLAUDE.md with any new agent capabilities
3. Test with the orchestrator workflow
4. Document input/output expectations

### Agent Guidelines

- One agent = one responsibility
- Use environment variables for paths
- Implement proper error handling
- Follow the scratchpad protocol for design decisions

## Reporting Issues

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Relevant log output

## Questions?

Feel free to open an issue for questions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
