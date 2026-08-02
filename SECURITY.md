# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.x (alpha) | ✅ |

## Reporting a vulnerability

We take security vulnerabilities seriously. Please report them by emailing the
maintainers directly — do not use public issue trackers for security issues.

**Please include:**

- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact

## Response timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 5 business days
- **Fix timeline**: Based on severity (critical: 7 days, high: 14 days, medium: 30 days)

## Disclosure policy

We follow coordinated disclosure:

1. Reporter submits vulnerability
2. Maintainers assess and develop fix
3. Fix is released
4. Vulnerability is publicly disclosed after release

## Security considerations for Atlas

As a framework for AI agent orchestration, be aware of:

- **Prompt injection**: Sanitize user input before passing to LLM providers
- **Data isolation**: KnowledgeBase documents are in-process memory by default
- **Plugin safety**: Third-party providers run in-process with full permissions
- **Dependency chain**: Review `pyproject.toml` for your chosen optional dependencies
