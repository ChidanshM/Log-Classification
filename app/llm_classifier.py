import os
import google.genai as genai
from typing import Tuple


class LLMClassifier:
    def __init__(self, model_name: str = "models/gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Set it in your environment.")

        # Create Gemini client
        self.client = genai.Client(api_key=api_key)
        self.model = model_name

    def predict(self, text: str) -> Tuple[str, float, str]:
        prompt = (
    "Classify the following log message into exactly one of these categories:\n"
    "authentication_failure, authentication_success, api_error, api_request, "
    "configuration_error, database_error, filesystem_error, network_error, "
    "resource_exhaustion, security_alert, service_timeout.\n\n"
    f"Log Message:\n{text}\n\n"
    "You must choose the closest category even if the classification is uncertain.\n"
    "Respond with ONLY valid JSON:\n"
    "{\n"
    "  \"label\": <category>,\n"
    "  \"confidence\": <0-1>,\n"
    "  \"explanation\": <short_reason>\n"
    "}\n"
)


        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        reply = response.text

        # Attempt to parse JSON inside LLM output
        import json,re
        label, conf, explanation = "unknown", 0.5, reply

        try:
            # Extract the first {...} block from the reply
            json_str = re.search(r"\{.*\}", reply, re.DOTALL).group(0)
        
            data = json.loads(json_str)
            label = data.get("label", label)
            conf = float(data.get("confidence", conf))
            explanation = data.get("explanation", explanation)
        
        except Exception:
            pass
        
        return label, conf, explanation
