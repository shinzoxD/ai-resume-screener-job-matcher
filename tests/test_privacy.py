from __future__ import annotations

from utils.privacy import redact_pii


def test_redact_pii_replaces_sensitive_values() -> None:
    text = """
    Contact me at jane.doe@email.com or +1 415-555-0112.
    LinkedIn: linkedin.com/in/janedoe
    GitHub: github.com/janedoe
    """
    redacted, counters = redact_pii(text)
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert counters["email"] == 1
    assert counters["phone"] >= 1

