from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

models = client.models.list()

print("\n=== Available Groq Models ===\n")
for m in models.data:
    print("-", m.id)
