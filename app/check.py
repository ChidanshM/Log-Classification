import os
import google.genai as genai
from google.genai.errors import ClientError

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
	exit(1)

client = genai.Client(api_key=api_key)

models = client.models.list()

supported = []

for m in models:
	name = m.name
	try:
		client.models.generate_content(model=name, contents="ping")
		supported.append(name)
	except ClientError as e:
		pass
	except Exception as e:
		pass

for s in supported:
	print(s)
