"""Deterministic text preprocessing pipeline implementing dual-stream architecture.

Preserves raw message text unchanged for structural/stylistic metadata extraction
while generating a parallel sanitized, normalized representation for lexical BoW modeling.
Supports both baseline lightweight sanitization and paper-described stemmed/stopword-filtered streams.
"""

from dataclasses import dataclass
import re
import string
from typing import Optional, Set, Union

# Regex for control characters (ASCII control characters except standard space)
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x1f\x7f-\x9f]")

# Regex for isolating URLs (matches http://, https://, or www.)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Multiple consecutive whitespace characters
MULTI_SPACE_PATTERN = re.compile(r"\s+")

# Lazy-loaded NLP tools for paper-stated stemmed preprocessing
_PORTER_STEMMER = None
_STOP_WORDS: Optional[Set[str]] = None


def _get_porter_stemmer():
    """Lazy loader for PorterStemmer."""
    global _PORTER_STEMMER
    if _PORTER_STEMMER is None:
        from nltk.stem import PorterStemmer

        _PORTER_STEMMER = PorterStemmer()
    return _PORTER_STEMMER


def _get_stop_words() -> Set[str]:
    """Lazy loader for standard English stopwords."""
    global _STOP_WORDS
    if _STOP_WORDS is None:
        try:
            from nltk.corpus import stopwords
            import nltk

            try:
                _STOP_WORDS = set(stopwords.words("english"))
            except LookupError:
                nltk.download("stopwords", quiet=True)
                _STOP_WORDS = set(stopwords.words("english"))
        except Exception:
            _STOP_WORDS = set()
    return _STOP_WORDS


@dataclass(frozen=True)
class DualStreamText:
    """Container holding parallel raw and sanitized text representations."""

    raw_text: str
    cleaned_text: str


def sanitize_text_for_lexical(
    text: str,
    lowercase: bool = True,
    remove_control_chars: bool = True,
    isolate_urls: bool = True,
    strip_punctuation: bool = False,
    remove_stopwords: bool = False,
    apply_stemming: bool = False,
) -> str:
    """Sanitize and normalize text for lexical vectorization (Stream 2).

    Applies deterministic sanitation:
    1. Trims leading and trailing whitespace.
    2. Excises non-printable control characters.
    3. Isolates embedded URLs.
    4. Converts characters to lowercase (if lowercase=True).
    5. Optionally strips punctuation (if strip_punctuation=True).
    6. Optionally removes stopwords and applies Porter stemming (Section III-C1 comparison).
    7. Collapses redundant consecutive whitespace.

    Args:
        text: Input string.
        lowercase: Whether to convert text to lowercase.
        remove_control_chars: Whether to remove ASCII control characters.
        isolate_urls: Whether to ensure URLs are bounded by whitespace.
        strip_punctuation: Whether to strip punctuation characters.
        remove_stopwords: Whether to filter English stop words.
        apply_stemming: Whether to apply Porter stemming.

    Returns:
        Sanitized string suitable for lexical vectorization.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 1. Leading/trailing whitespace
    s = text.strip()

    # 2. Control characters excision
    if remove_control_chars:
        s = CONTROL_CHARS_PATTERN.sub(" ", s)

    # 3. Ensure URLs have whitespace padding around them
    if isolate_urls:
        s = URL_PATTERN.sub(lambda m: f" {m.group(0)} ", s)

    # 4. Lowercasing
    if lowercase:
        s = s.lower()

    # 5. Optional punctuation stripping
    if strip_punctuation:
        s = s.translate(str.maketrans("", "", string.punctuation))

    # 6. Optional stopword removal and Porter stemming (Section III-C1)
    if remove_stopwords or apply_stemming:
        tokens = s.split()
        if remove_stopwords:
            sw = _get_stop_words()
            tokens = [t for t in tokens if t not in sw]
        if apply_stemming:
            stemmer = _get_porter_stemmer()
            tokens = [stemmer.stem(t) for t in tokens]
        s = " ".join(tokens)

    # 7. Normalization of whitespace
    s = MULTI_SPACE_PATTERN.sub(" ", s).strip()
    return s


def preprocess_message(
    message: Union[str, None],
    lowercase: bool = True,
    remove_control_chars: bool = True,
    isolate_urls: bool = True,
    strip_punctuation: bool = False,
    remove_stopwords: bool = False,
    apply_stemming: bool = False,
) -> DualStreamText:
    """Process an incoming message into dual streams.

    Stream 1 (raw_text): Unaltered raw string for metadata feature extraction.
    Stream 2 (cleaned_text): Sanitized string for lexical vectorization.

    Args:
        message: Raw input SMS string.
        lowercase: Whether to lowercase cleaned_text.
        remove_control_chars: Whether to excise control characters.
        isolate_urls: Whether to isolate URLs with whitespace.
        strip_punctuation: Whether to strip punctuation characters.
        remove_stopwords: Whether to filter English stop words.
        apply_stemming: Whether to apply Porter stemming.

    Returns:
        DualStreamText object.
    """
    raw = "" if message is None else str(message)
    cleaned = sanitize_text_for_lexical(
        raw,
        lowercase=lowercase,
        remove_control_chars=remove_control_chars,
        isolate_urls=isolate_urls,
        strip_punctuation=strip_punctuation,
        remove_stopwords=remove_stopwords,
        apply_stemming=apply_stemming,
    )
    return DualStreamText(raw_text=raw, cleaned_text=cleaned)
