# app/api.py

import csv
import io
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.router import HybridClassifier
from app.models import LogRequest, LogResponse, BatchLogResponseItem


app = FastAPI(title="Hybrid Log Classification API")

# CORS for frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = HybridClassifier()


# ---------------------------------------------------------
# Single Log Classification
# ---------------------------------------------------------
@app.post("/classify", response_model=LogResponse)
def classify_log(request: LogRequest):
    result = classifier.classify(request.log)

    return LogResponse(
        label=result["label"],
        confidence=result["confidence"],
        layer=result["layer"],
        llm_explanation=result.get("llm_explanation"),
    )


# ---------------------------------------------------------
# Batch Classification (CSV upload)
# ---------------------------------------------------------
@app.post("/classify_csv", response_model=List[BatchLogResponseItem])
async def classify_csv(file: UploadFile = File(...), log_column: str = "log"):

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        content = await file.read()
        text = io.StringIO(content.decode("utf-8"))
        reader = csv.DictReader(text)
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to parse CSV file.")

    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV file is empty or invalid.")

    if log_column not in reader.fieldnames:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain a '{log_column}' column. Found: {reader.fieldnames}",
        )

    results: List[BatchLogResponseItem] = []

    for line_no, row in enumerate(reader, start=1):
        log = row.get(log_column, "")

        # Safe processing for each line
        try:
            result = classifier.classify(log)
        except Exception as e:
            # Log error and continue
            result = {
                "label": "error",
                "confidence": 0.0,
                "layer": "none",
                "llm_explanation": f"Error processing line: {str(e)}"
            }

        results.append(
            BatchLogResponseItem(
                line_number=line_no,
                log=log,
                label=result["label"],
                confidence=result["confidence"],
                layer=result["layer"],
                llm_explanation=result.get("llm_explanation"),
            )
        )

    # FastAPI automatically serializes list as JSON
    return results
