"""Unit tests for dual-stream deterministic text preprocessing."""

import pytest
from app.ml.preprocessing import preprocess_message, sanitize_text_for_lexical


def test_sanitize_whitespace_and_casing():
    """Verify whitespace trimming, multi-space collapse, and lowercasing."""
    raw = "   HELLO    WORLD!  Welcome to   SMS.   "
    clean = sanitize_text_for_lexical(raw)
    assert clean == "hello world! welcome to sms."


def test_control_character_removal():
    """Verify non-printable control characters are excised."""
    raw = "Hello\x00\x07World\x1fTest"
    clean = sanitize_text_for_lexical(raw)
    assert "\x00" not in clean
    assert "\x07" not in clean
    assert "\x1f" not in clean
    assert clean == "hello world test"


def test_url_isolation():
    """Verify URLs are isolated with whitespace padding."""
    raw = "Visit:http://spam.org/claim!Free stuff"
    clean = sanitize_text_for_lexical(raw, isolate_urls=True)
    assert "http://spam.org/claim" in clean


def test_dual_stream_separation():
    """Verify that raw text is strictly preserved untouched in Stream 1."""
    raw = "  URGENT! Call 08452810075 Now for £500 Voucher: http://claim.com  "
    stream = preprocess_message(raw)

    # Stream 1: Exact verbatim original
    assert stream.raw_text == raw
    assert "URGENT!" in stream.raw_text
    assert "£500" in stream.raw_text

    # Stream 2: Sanitized and normalized
    assert stream.cleaned_text.startswith("urgent!")
    assert "call 08452810075" in stream.cleaned_text


def test_empty_and_null_inputs():
    """Verify safe handling of None and empty strings."""
    s1 = preprocess_message("")
    assert s1.raw_text == ""
    assert s1.cleaned_text == ""

    s2 = preprocess_message(None)
    assert s2.raw_text == ""
    assert s2.cleaned_text == ""

    s3 = preprocess_message("    ")
    assert s3.cleaned_text == ""
