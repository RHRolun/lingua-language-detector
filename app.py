from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
from lingua import Language, LanguageDetectorBuilder

app = FastAPI(title="Lingua Language Detector")

# Build detector once at startup (detects all languages)
detector = LanguageDetectorBuilder.from_all_languages().build()


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
    evidences: Optional[List] = None
    metadata: Optional[Dict[str, Any]] = None


def detect_languages(text: str, allowed_languages: List[str]) -> List[ContentAnalysisResponse]:
    """Detect non-allowed language segments in text."""
    results = detector.detect_multiple_languages_of(text)
    detections = []

    for result in results:
        lang_name = result.language.name
        if lang_name.upper() not in [lang.upper() for lang in allowed_languages]:
            snippet = text[result.start_index:result.end_index]
            confidence = detector.compute_language_confidence(text[result.start_index:result.end_index], result.language)
            detections.append(ContentAnalysisResponse(
                start=result.start_index,
                end=result.end_index,
                text=snippet,
                detection=lang_name,
                detection_type="language",
                score=confidence,
                evidences=None,
                metadata={"language": lang_name}
            ))

    return detections


@app.get("/health")
def health():
    return "ok"


@app.post("/api/v1/text/contents", response_model=List[List[ContentAnalysisResponse]])
def analyze_contents(request: ContentAnalysisHttpRequest):
    """Analyze text contents for non-allowed languages."""
    # Default to English only
    allowed_languages = ["ENGLISH"]
    if request.detector_params and "allowed_languages" in request.detector_params:
        allowed_languages = request.detector_params["allowed_languages"]

    response = []
    for content in request.contents:
        detections = detect_languages(content, allowed_languages)
        response.append(detections)

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
