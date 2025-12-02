import csv
import io
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.router import HybridClassifier
from app.models import LogRequest, LogResponse, BatchLogResponseItem


app = FastAPI(title="Log Processing & Classification Service (v1)")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_methods=["*"],
	allow_headers=["*"],
)

log_analyzer = HybridClassifier()

# (V1) Single Log Analysis
@app.post("/v1/logs/analyze", response_model=LogResponse)
def analyze_single_log(payload: LogRequest):
	result = log_analyzer.label_logs(payload.log)

	return LogResponse(
		label=result["label"],
		confidence=result["confidence"],
		layer=result["layer"],
		llm_explanation=result.get("llm_explanation"),
	)


# (V1) Batch Classification (CSV upload)
@app.post("/v1/logs/analyze/batch", response_model=List[BatchLogResponseItem])
async def analyze_csv_batch(file: UploadFile = File(...), log_column: str = "log"):

	if not file.filename.endswith(".csv"):
		raise HTTPException(status_code=400, detail="Only CSV files are supported.")

	try:
		content = await file.read()
		decoded_text = io.StringIO(content.decode("utf-8"))
		csv_reader = csv.DictReader(decoded_text)
	except Exception:
		raise HTTPException(status_code=400, detail="Unable to read CSV file.")

	if not csv_reader.fieldnames:
		raise HTTPException(status_code=400, detail="CSV header is empty or invalid.")

	if log_column not in csv_reader.fieldnames:
		raise HTTPException(
			status_code=400,
			detail=f"CSV must contain a '{log_column}' column. Found: {csv_reader.fieldnames}",
		)

	batch_results: List[BatchLogResponseItem] = []

	for line_no, row in enumerate(csv_reader, start=1):
		log = row.get(log_column, "")

	   
		try:
			result = log_analyzer.label_logs(log)
		except Exception as e:
			
			result = {
				"label": "error",
				"confidence": 0.0,
				"layer": "none",
				"llm_explanation": f"Error processing line: {str(e)}"
			}

		batch_results.append(
			BatchLogResponseItem(
				line_number=line_no,
				log=log,
				label=result["label"],
				confidence=result["confidence"],
				layer=result["layer"],
				llm_explanation=result.get("llm_explanation"),
			)
		)


	return batch_results
