# app/router.py

from typing import Dict, Any

from app.log_parser import LogParser
from app.regex_classifier import RegexClassifier
from app.ml_classifier import MLClassifier
from app.llm_classifier import LLMClassifier
from app.config import settings


class HybridClassifier:
	"""
	Hybrid classification pipeline:
	1. Regex (fast, deterministic)
	2. ML classifier (probabilistic)
	3. LLM fallback (slow, most accurate)
	"""

	def __init__(self):
		self.parser = LogParser()
		self.regex = RegexClassifier()
		self.ml = MLClassifier()
		self.llm = LLMClassifier()

	def label_logs(self, raw_log: str) -> Dict[str, Any]:
		# Parse log 
		parsed = self.parser.parse(raw_log)
		message = parsed.get("message", raw_log)

		# Regex stage 
		r_label, r_conf = self._safe_regex(message)
		if r_label is not None and r_conf >= settings.REGEX_CONFIDENCE:
			return {
				"label": r_label,
				"confidence": r_conf,
				"layer": "regex",
				"parsed": parsed,
			}

		# ML stage 
		ml_label, ml_conf = self.ml.predict(message)
		print("ML Label: ", ml_label)
		print("ML Confidence: ", ml_conf)
		if ml_conf >= settings.ML_CONFIDENCE:
			return {
				"label": ml_label,
				"confidence": ml_conf,
				"layer": "ml",
				"parsed": parsed,
			}

		# LLM fallback 
		llm_label, llm_conf, explanation = self.llm.predict(message)
		return {
			"label": llm_label,
			"confidence": llm_conf,
			"layer": "llm",
			"parsed": parsed,
			"llm_explanation": explanation,
		}

	# Internal helper: regex safety wrapper
	def _safe_regex(self, message: str):
		"""
		Ensures regex classifier returns (label, confidence)
		even if implemented without confidence.
		"""
		result = self.regex.predict(message)

		# If classifier only returns label
		if isinstance(result, str):
			return result, 1.0

		# If it returned (label, confidence)
		if isinstance(result, tuple) and len(result) == 2:
			return result[0], float(result[1])

		# Otherwise: no match
		return None, 0.0
