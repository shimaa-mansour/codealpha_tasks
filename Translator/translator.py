"""Translation logic using Google Translate via the deep-translator library."""
from deep_translator import GoogleTranslator

MAX_CHARS = 5000


def get_languages() -> dict:
    """Return {language name: language code}, e.g. {'english': 'en'}."""
    return GoogleTranslator().get_supported_languages(as_dict=True)


def translate_text(text: str, source: str, target: str) -> str:
    """Send text to Google Translate and return the translated text."""
    text = text.strip()
    if not text:
        raise ValueError("Please enter some text to translate.")
    if len(text) > MAX_CHARS:
        raise ValueError(f"Text is too long (max {MAX_CHARS} characters).")
    if source == target:
        return text
    return GoogleTranslator(source=source, target=target).translate(text)
