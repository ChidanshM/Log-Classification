# app/models.py

from typing import Optional
from pydantic import BaseModel


class LogRequest(BaseModel):
    """Single-log API request."""
    log: str


class LogResponse(BaseModel):
    """Single-log API response."""
    label: str
    confidence: float
    layer: str
    llm_explanation: Optional[str] = None


class BatchLogResponseItem(BaseModel):
    """Response item for CSV batch processing."""
    line_number: int
    log: str
    label: str
    confidence: float
    layer: str
    llm_explanation: Optional[str] = None
