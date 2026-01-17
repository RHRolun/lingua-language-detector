import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from lingua import Language, LanguageDetectorBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lingua Language Detector")

# Exclude languages that cause false positives on short English text
EXCLUDED_LANGUAGES = [
    Language.SHONA,
    Language.XHOSA,
    Language.SOTHO,
    Language.TSONGA,
    Language.TSWANA,
    Language.GANDA,
]

# Build detector from all languages except excluded ones
supported_languages = [lang for lang in Language.all() if lang not in EXCLUDED_LANGUAGES]
detector = LanguageDetectorBuilder.from_languages(*supported_languages).build()

# Minimum confidence threshold for a detection to be considered valid
MIN_CONFIDENCE_THRESHOLD = 0.1


# Request/Response schemas matching guardrails-detectors
class ContentAnalysisHttpRequest(BaseModel):
    contents: List[str] = Field(min_length=1)
    detector_params: Optional[Dict[str, Any]] = None


class ContentAnalysisResponse(BaseModel):
    start: int
    end: int
    text: str
    detection: str
    detection_type: str
    score: float
    evidences: List[Any] = []
    metadata: Dict[str, Any] = {}


def detect_language(text: str) -> List[ContentAnalysisResponse]:
    """
    Detect the primary language of text.
    Returns empty list if text is English.
    Returns detection only if text is non-English.
    """
    if not text or not text.strip():
        logger.debug(f"Empty text, skipping detection")
        return []

    # Get confidence values for all languages to understand the detection
    confidence_values = detector.compute_language_confidence_values(text)
    top_languages = sorted(confidence_values, key=lambda x: x.value, reverse=True)[:5]
    logger.info(f"Text: '{text}' | Top languages: {[(l.language.name, round(l.value, 3)) for l in top_languages]}")

    # Detect the primary language
    detected_lang = detector.detect_language_of(text)
    logger.info(f"Detected language: {detected_lang.name if detected_lang else 'None'}")

    # If can't detect, allow it
    if detected_lang is None:
        logger.info(f"No language detected")
        return []

    # Get confidence scores
    english_confidence = detector.compute_language_confidence(text, Language.ENGLISH)
    detected_confidence = detector.compute_language_confidence(text, detected_lang)

    # If detected language confidence is below threshold, treat as uncertain
    if detected_confidence < MIN_CONFIDENCE_THRESHOLD:
        logger.info(f"Detected {detected_lang.name} but confidence {detected_confidence:.3f} < {MIN_CONFIDENCE_THRESHOLD}, ignoring")
        return []

    # If English, allow it
    if detected_lang == Language.ENGLISH:
        logger.info(f"English detected (conf: {english_confidence:.3f}), allowing")
        return []

    # Non-English detected with sufficient confidence - return detection
    # Score = how "non-English" the text is (higher = more likely to block)
    score = 1.0 - english_confidence

    logger.info(f"Non-English detected: {detected_lang.name} (conf: {detected_confidence:.3f}) vs English (conf: {english_confidence:.3f})")

    return [ContentAnalysisResponse(
        start=0,
        end=len(text),
        text=text,
        detection="non_english",
        detection_type="language_detection",
        score=score,
        evidences=[],
        metadata={
            "detected_language": detected_lang.name,
            "english_confidence": english_confidence
        }
    )]


@app.get("/health")
def health():
    return "ok"


@app.post("/api/v1/text/contents", response_model=List[List[ContentAnalysisResponse]])
def analyze_contents(request: ContentAnalysisHttpRequest):
    """
    Analyze text contents for language detection.
    Returns empty array for each content that is English.
    Returns detection for non-English content.
    """
    logger.info(f"Received request: contents={request.contents}, params={request.detector_params}")

    response = []
    for content in request.contents:
        detections = detect_language(content)
        response.append(detections)

    logger.info(f"Returning response: {response}")
    return response


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(path: str, request: Request):
    """Catch-all to log any unhandled routes."""
    body = await request.body()
    logger.warning(f"Unhandled route: {request.method} /{path} - body: {body}")
    return {"error": f"Unknown endpoint: /{path}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
