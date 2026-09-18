"""Text cleaning, section detection, and page-aware chunking utilities."""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """Represents a text chunk with associated metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# Agricultural topic keywords mapped to canonical topic identifiers
TOPIC_PATTERNS = [
    (r"\b(botanical|kashayam|asthra|neem|nske|dashaparni|jeevamrit|beejamrit|agniastra|brahmastra|neemastra|extract)\b", "botanical_preparations"),
    (r"\b(biological|trichogramma|chrysoperla|predator|parasitoid|pseudomonas|trichoderma|bio-agent|beneficial insect)\b", "biological_management"),
    (r"\b(cultural|sanitation|field sanitation|crop rotation|intercrop|intercropping|mulch|mulching|crop diversification|trap crop|border crop)\b", "cultural_practices"),
    (r"\b(mechanical|physical|light trap|sticky trap|pheromone|bird perch|hand picking|trenches)\b", "mechanical_practices"),
    (r"\b(disease|blight|wilt|rot|mildew|rust|smut|canker|anthracnose)\b", "disease_management"),
    (r"\b(pest|insect|borer|bollworm|aphid|whitefly|thrips|hopper|jassid|caterpillar|mite)\b", "pest_management"),
]

# Section title matching rules
SECTION_PATTERNS = [
    (r"^(preparation of|botanical|kashayam|asthra|neem|nske|dashaparni|jeevamrut|beejamrut|agniastra|brahmastra|neemastra).*", "Botanical Preparations"),
    (r".*(biological control|biological management|predator|parasitoid|bio-agent).*", "Biological Management"),
    (r".*(cultural practice|sanitation|field sanitation|intercropping|mulching|crop diversification).*", "Cultural Practices"),
    (r".*(mechanical control|mechanical practice|pheromone trap|light trap|sticky trap).*", "Mechanical Practices"),
    (r".*(pest and disease management|pest & disease|disease management).*", "Pest and Disease Management"),
    (r".*(pest management|insect management|pest identification|symptom).*", "Pest Management"),
    (r"^(chapter|module|section|unit)\s+\d+[:\s\-]*(.+)", r"\2"),
]


def clean_text(text: str) -> str:
    """Normalize whitespace and strip extraneous leading/trailing spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def detect_section(line: str) -> Optional[str]:
    """Detect if a text line represents a section or chapter heading."""
    clean_line = line.strip()
    if not clean_line or len(clean_line) > 120:
        return None

    # Check for strong section heading matches
    for pattern, label in SECTION_PATTERNS:
        match = re.search(pattern, clean_line, re.IGNORECASE)
        if match:
            if "\\" in label:
                return clean_line.strip()
            return label

    # Check for short uppercase or title-case lines that look like headings
    words = clean_line.split()
    if 1 <= len(words) <= 8 and (clean_line.isupper() or clean_line.istitle()):
        if not clean_line.endswith((".", ",", ";", ":")):
            return clean_line.title()

    return None


def detect_topic(text: str, default: str = "natural_farming_practices") -> str:
    """Detect the most relevant agricultural topic from the text."""
    lower_text = text.lower()
    for pattern, topic in TOPIC_PATTERNS:
        if re.search(pattern, lower_text):
            return topic
    return default


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """Split clean text into overlapping word-based chunks.
    
    Raises:
        ValueError: If overlap is negative or greater than/equal to chunk_size.
    """
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"Invalid overlap ({overlap}): overlap must be >= 0 and strictly less than chunk_size ({chunk_size})"
        )

    cleaned = clean_text(text)
    if not cleaned:
        return []

    words = cleaned.split()
    if not words:
        return []

    meta = dict(metadata or {})
    chunks: List[Chunk] = []
    step = chunk_size - overlap

    for start_idx in range(0, len(words), step):
        chunk_words = words[start_idx : start_idx + chunk_size]
        chunk_str = " ".join(chunk_words)
        chunks.append(Chunk(text=chunk_str, metadata=meta.copy()))
        if start_idx + chunk_size >= len(words):
            break

    return chunks


def chunk_page_text(
    page_text: str,
    page_number: int,
    base_metadata: Dict[str, Any],
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """Split page text into section-aware chunks preserving page and section metadata.
    
    Ensures headings stay attached to subsequent paragraphs and chunks approximately
    400-700 tokens (~300-550 words) with 50-100 token overlap.
    """
    if not page_text or not page_text.strip():
        return []

    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return []

    # Identify primary section from the page or header lines
    current_section = base_metadata.get("section", "General Advisory")
    for line in lines[:5]:
        sec = detect_section(line)
        if sec:
            current_section = sec
            break

    # Build coherent paragraphs while preserving headings
    paragraphs: List[str] = []
    curr_para: List[str] = []

    for line in lines:
        sec = detect_section(line)
        if sec:
            if curr_para:
                paragraphs.append(" ".join(curr_para))
                curr_para = []
            current_section = sec
            curr_para.append(f"[{sec}] {line}")
        else:
            curr_para.append(line)

    if curr_para:
        paragraphs.append(" ".join(curr_para))

    full_page_clean = " ".join(paragraphs)
    words = full_page_clean.split()
    if not words:
        return []

    chunks: List[Chunk] = []
    step = max(1, chunk_size - overlap)

    for start_idx in range(0, len(words), step):
        chunk_words = words[start_idx : start_idx + chunk_size]
        chunk_str = " ".join(chunk_words)

        # Detect specific topic for this chunk or fall back to document base topic
        detected_topic = detect_topic(chunk_str, default=base_metadata.get("topic", "natural_farming_practices"))

        chunk_meta = dict(base_metadata)
        chunk_meta["page"] = page_number
        chunk_meta["section"] = current_section
        chunk_meta["topic"] = detected_topic

        chunks.append(Chunk(text=chunk_str, metadata=chunk_meta))
        if start_idx + chunk_size >= len(words):
            break

    return chunks
