import logging
from src.agents.llm_client import LLMClient
from src.config import (
    OPENAI_API_TYPE,
    OPENAI_API_KEY,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    OPENAI_API_VERSION,
    EMBED_MODEL,
    AZURE_OPENAI_DEPLOYMENT_NAME
)


class SummaryAgent:
    def __init__(self, max_tokens=100):
        if OPENAI_API_TYPE == "azure":
            self.llm_client = LLMClient(
                provider="azure",
                api_key=AZURE_OPENAI_API_KEY,
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=OPENAI_API_VERSION
            )
        else:
            self.llm_client = LLMClient(
                provider="openai",
                api_key=OPENAI_API_KEY,
                model=AZURE_OPENAI_DEPLOYMENT_NAME # uses same chatbot
            )
        self.max_tokens = max_tokens

    def summarize(self, text: str) -> str:
        try:
            prompt = (
                "Resume o texto abaixo em 2-3 frases claras e objetivas, destacando as principais regras e orientações relevantes para arquitetos e construtores sobre edificações urbanas:\n\n"
                f"{text}\n\nResumo:"
            )
            return self.llm_client.run(prompt, max_tokens=self.max_tokens)
        except Exception as e:
            logging.error(f"Summary generation failed: {e}")
            return ""
