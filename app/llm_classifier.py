import os
import json
import re
import hashlib
import time
from typing import Tuple, Dict, Optional
import google.genai as genai


# Simple in-memory cache for LLM responses
_llm_cache: Dict[str, Tuple[str, float, str]] = {}


class LLMClassifier:
	"""
	Enhanced LLM-based classifier with improvements:
	- Few-shot prompting with examples
	- Response caching to reduce API calls
	- Retry logic with exponential backoff
	- Better structured output parsing
	- Confidence calibration
	"""
	
	def __init__(self, model_name: str = "models/gemini-2.5-flash-lite"):
		api_key = os.getenv("GEMINI_API_KEY")

		if not api_key:
			raise ValueError("GEMINI_API_KEY is missing. Set it in your environment.")

		# Create Gemini client
		self.client = genai.Client(api_key=api_key)
		self.model = model_name
		self.max_retries = 3
		self.cache = _llm_cache

	def _create_few_shot_prompt(self, text: str) -> str:
		"""
		Create an enhanced prompt with few-shot examples for better accuracy.
		"""
		prompt = """You are an expert log classification system. Classify the following log message into EXACTLY ONE category.

# CATEGORIES:
1. authentication_failure - Failed login attempts, invalid credentials, token failures
2. authentication_success - Successful logins, token validation, session establishment
3. api_error - API crashes, malformed requests, HTTP 4xx/5xx errors in API calls
4. api_request - Normal API requests (GET, POST, PUT, DELETE, etc.)
5. configuration_error - Missing config keys, invalid YAML/JSON, parameter errors
6. database_error - DB connection failures, SQL errors, transaction deadlocks
7. filesystem_error - Disk quota, file not found, permission denied, corruption
8. network_error - DNS failures, connection resets, packet loss, network timeouts
9. resource_exhaustion - Out of memory, CPU starvation, thread pool exhaustion
10. security_alert - Unauthorized access, malware, suspicious behavior, firewall blocks
11. service_timeout - Service timeouts, request timeouts, operation timeouts

# EXAMPLES:

Input: "Failed login attempt for user 'admin'"
Output: {{"label": "authentication_failure", "confidence": 0.98, "explanation": "Clear failed login pattern"}}

Input: "User 'john' successfully authenticated"
Output: {{"label": "authentication_success", "confidence": 0.99, "explanation": "Successful authentication indicated"}}

Input: "API error 500 for endpoint /v1/process"
Output: {{"label": "api_error", "confidence": 0.97, "explanation": "HTTP 500 error in API call"}}

Input: "GET /v1/metrics from client"
Output: {{"label": "api_request", "confidence": 0.95, "explanation": "Normal GET request"}}

Input: "Configuration error: missing key 'cluster.id'"
Output: {{"label": "configuration_error", "confidence": 0.98, "explanation": "Missing configuration key"}}

Input: "Database connection refused on 10.0.0.5"
Output: {{"label": "database_error", "confidence": 0.99, "explanation": "Database connection failure"}}

Input: "Out of memory: process 'worker-1' killed"
Output: {{"label": "resource_exhaustion", "confidence": 0.99, "explanation": "Memory exhaustion causing process kill"}}

Input: "Security alert: unauthorized access attempt by 'hacker'"
Output: {{"label": "security_alert", "confidence": 0.99, "explanation": "Unauthorized access attempt"}}

Input: "Network unreachable while contacting 10.0.0.1"
Output: {{"label": "network_error", "confidence": 0.98, "explanation": "Network connectivity issue"}}

Input: "Request timed out contacting 'auth-service'"
Output: {{"label": "service_timeout", "confidence": 0.97, "explanation": "Service request timeout"}}

# TASK:
Classify this log message:

Input: "{log_message}"

Respond with ONLY valid JSON in this exact format:
{{"label": "<category>", "confidence": <0.0-1.0>, "explanation": "<brief reason>"}}

Choose the MOST SPECIFIC category. If uncertain, use lower confidence (0.6-0.8).
"""
		return prompt.format(log_message=text)

	def _get_cache_key(self, text: str) -> str:
		"""Generate cache key from log text."""
		return hashlib.md5(text.encode('utf-8')).hexdigest()

	def predict(self, text: str) -> Tuple[str, float, str]:
		"""
		Predict log category using LLM with caching and retry logic.
		
		Args:
			text: Log message to classify
			
		Returns:
			(label, confidence, explanation)
		"""
		# Check cache first
		cache_key = self._get_cache_key(text)
		if cache_key in self.cache:
			return self.cache[cache_key]
		
		# Prepare prompt
		prompt = self._create_few_shot_prompt(text)
		
		# Retry logic with exponential backoff
		for attempt in range(self.max_retries):
			try:
				response = self.client.models.generate_content(
					model=self.model,
					contents=prompt
				)
				
				reply = response.text.strip()
				print(reply)
				# Parse JSON response
				label, conf, explanation = self._parse_response(reply)
				
				# Cache the result
				result = (label, conf, explanation)
				self.cache[cache_key] = result
				
				return result
				
			except Exception as e:
				if attempt < self.max_retries - 1:
					wait_time = 2 ** attempt  
					time.sleep(wait_time)
				else:
					return "unknown", 0.3, f"API Error: {str(e)}"
	
	def _parse_response(self, reply: str) -> Tuple[str, float, str]:
		"""
		Parse LLM response and extract label, confidence, and explanation.
		"""
		label, conf, explanation = "unknown", 0.5, reply
		
		try:
			# Try to extract JSON block
			json_match = re.search(r'\{.*\}', reply, re.DOTALL)
			if json_match:
				json_str = json_match.group(0)
				data = json.loads(json_str)
				
				label = data.get("label", label)
				conf = float(data.get("confidence", conf))
				explanation = data.get("explanation", explanation)
				
				# Confidence bounds
				conf = max(0.0, min(1.0, conf))
			else:
				# Fallback: try to extract label from text
				for category in [
					"authentication_failure", "authentication_success",
					"api_error", "api_request", "configuration_error",
					"database_error", "filesystem_error", "network_error",
					"resource_exhaustion", "security_alert", "service_timeout"
				]:
					if category in reply.lower():
						label = category
						conf = 0.7
						break
		
		except json.JSONDecodeError:
			 
			pass
		except Exception as e:
			pass
		
		return label, conf, explanation
	
	def clear_cache(self):
		"""Clear the response cache."""
		self.cache.clear()
	
	def get_cache_stats(self) -> Dict[str, int]:
		"""Get cache statistics."""
		return {
			"cache_size": len(self.cache),
			"cache_hits": 0 
		}

