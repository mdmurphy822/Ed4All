# Security Policy

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **GitHub Security Advisories (Preferred)**: Use [GitHub's private vulnerability reporting](https://github.com/[your-org]/courseforge/security/advisories/new) to submit a report directly.

2. **Email**: If you prefer email, contact the maintainers directly (see repository for contact information).

### What to Include

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Any suggested fixes (optional)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Target**: Depends on severity, typically within 30 days for critical issues

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |

## Security Considerations

When using Courseforge, be aware of the following security considerations:

### File Processing

- **IMSCC Packages**: Courseforge processes IMSCC packages which may contain HTML, XML, and embedded files. Always validate packages from untrusted sources.
- **XML Parsing**: The system uses secure XML parsing with external entity (XXE) protection enabled.
- **File Uploads**: Input files should be validated before processing through the intake pipeline.

### Environment Configuration

- **Environment Variables**: Sensitive paths like `DART_PATH` should be set via environment variables, never hardcoded.
- **Output Directories**: The `exports/` directory contains generated content. Ensure appropriate access controls in production environments.

### Dependencies

- All dependencies are regularly monitored for security updates.
- Run `pip install --upgrade -r scripts/requirements.txt` periodically to get security patches.

## Disclosure Policy

We follow responsible disclosure practices:

1. Security issues are addressed before public disclosure
2. Credit is given to reporters (unless anonymity is requested)
3. A security advisory is published after the fix is released
