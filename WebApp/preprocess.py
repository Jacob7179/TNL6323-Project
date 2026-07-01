import re
import emoji

_ASPECT_RATING = re.compile(
    r"(food|service|atmosphere|ambien?ce)\s*:?\s*\d(?:\.\d)?\s*/\s*5",
    re.IGNORECASE,
)
_META_MARKERS = [
    r"Group size", r"Wait time", r"Seating type", r"Price per person",
    r"Reservation recommended", r"Service\s*:\s*Dine in", r"Meal type",
    r"Recommended dishes", r"Parking space", r"Parking options",
    r"Dietary options", r"Kid-friendliness", r"Wheelchair",
]
_META_RE = re.compile("|".join(_META_MARKERS), re.IGNORECASE)

_URL = re.compile(r"https?://\S+|www\.\S+")
_MULTISPACE = re.compile(r"\s+")
# tick / check symbols sometimes left from the rating widget
_NOISE_SYMBOLS = re.compile(r"[✓✔|•]+")


def strip_metadata(text: str) -> str:
    """Remove trailing Google Maps metadata and embedded aspect ratings."""
    if not isinstance(text, str):
        return ""
    text = _ASPECT_RATING.sub(" ", text)
    m = _META_RE.search(text)
    if m:
        text = text[: m.start()]
    return text


def clean_text(text: str, demojize: bool = True) -> str:
    """
    Full cleaning pipeline used for model input.

    demojize=True converts 😍 -> ':smiling_face_with_heart_eyes:' so emoji
    sentiment survives into the bag-of-words features.
    """
    if not isinstance(text, str):
        return ""
    text = strip_metadata(text)
    text = _URL.sub(" ", text)
    text = _NOISE_SYMBOLS.sub(" ", text)
    if demojize:
        # turn emojis into words, using spaces as delimiters
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = text.replace("_", " ")
    text = text.lower()
    text = _MULTISPACE.sub(" ", text).strip()
    return text


def extract_emojis(text: str):
    """Return the list of emoji characters present in the raw text."""
    if not isinstance(text, str):
        return []
    return [e["emoji"] for e in emoji.emoji_list(text)]


if __name__ == "__main__":
    sample = ("Food tasted good, staff also very attentive 😍 "
              "Food: 5/5 | Service: 5/5 | Atmosphere: 5/5 "
              "Group size 3-4 people Wait time Up to 10 min")
    print("RAW :", sample)
    print("META:", strip_metadata(sample))
    print("CLEAN:", clean_text(sample))
    print("EMOJI:", extract_emojis(sample))
