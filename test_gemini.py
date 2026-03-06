from langchain_google_genai import ChatGoogleGenerativeAI
import time

GOOGLE_API_KEY = "AIzaSyBnLUdGRPVSntf2iU_PE8Qm5UrGVXywDg0"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GOOGLE_API_KEY,
)

print("🚀 Testing Gemini 2.0 Flash Lite...")
try:
    response = llm.invoke("Hello, are you available?")
    print(f"✅ Success: {response.content}")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    if "429" in str(e) or "quota" in str(e).lower():
        print("💡 Confirmed: Rate limit is active.")
