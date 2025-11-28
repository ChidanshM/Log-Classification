import os
import google.genai as genai
from google.genai.errors import ClientError

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in environment")
    exit(1)

client = genai.Client(api_key=api_key)

print("✅ Gemini client initialized!\n")
print("📌 Checking which models support generate_content...\n")

models = client.models.list()

supported = []

for m in models:
    name = m.name
    try:
        # Test with an empty prompt (cheap capability check)
        client.models.generate_content(model=name, contents="ping")
        supported.append(name)
        print(f"✔ {name} supports generate_content")
    except ClientError as e:
        # Model does not support generate_content
        print(f"✖ {name} does NOT support generate_content ({e.status})")
    except Exception as e:
        print(f"⚠ Error checking {name}: {e}")

print("\n🎯 Models that support content generation:")
for s in supported:
    print(f"➡ {s}")
