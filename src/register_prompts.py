import mlflow
import os
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# --- Define Prompts to be Registered ---
PROMPTS_TO_REGISTER = {
    "legal_expert": {
        "template": """
        You are an AI assistant specialized in Portuguese building regulations.
        Use the following context to answer the question.
        If you don't know the answer from the context, state that clearly.
        Provide a concise and direct answer based strictly on the provided text. Cite the article number if possible.
        Context: {context}
        Question: {question}
        Answer (in Portuguese):
        """,
        "tags": {"use_case": "legal_chatbot", "persona": "expert"}
    },
    "homeowner_guide": {
        "template": """
        You are a helpful AI assistant for homeowners in Portugal.
        Use the provided legal information to answer the question in a simple, easy-to-understand way.
        Explain the key points without complex legal jargon.
        Based on the regulations: {context}
        Question: {question}
        Helpful Answer (in Portuguese):
        """,
        "tags": {"use_case": "legal_chatbot", "persona": "guide"}
    }
}

def register_prompts():
    print("📝 Starting prompt registration in MLflow...")
    for name, config in PROMPTS_TO_REGISTER.items():
        try:
            mlflow.register_prompt(
                name=name,
                template=config["template"],
                tags=config["tags"]
            )
            print(f"✅ Successfully registered prompt: '{name}'")
        except Exception as e:
            print(f"⚠️  Could not register prompt '{name}'. It might already exist. Error: {e}")

if __name__ == "__main__":
    register_prompts()
    print("\n🎉 Prompt registration process finished.")
    print(f"Check the 'Prompts' tab in the MLflow UI: {MLFLOW_TRACKING_URI.replace(':5000', ':5001')}")
