import google.generativeai as genai
import os

api_key = "AIzaSyDr1T1yV84WTJ6CT8KMw63PtobiNaV43z0"
genai.configure(api_key=api_key)

try:
    print("Listing available models...")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
