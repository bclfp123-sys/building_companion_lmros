from openai import AzureOpenAI, OpenAI

class LLMClient:
    def __init__(
        self,
        provider: str = "openai",   # "openai" or "azure"
        api_key: str = None,
        model: str = None,
        endpoint: str = None,
        api_version: str = "2024-05-01-preview"
    ):
        self.provider = provider.lower()

        if self.provider == "azure":
            if not all([api_key, endpoint, model]):
                raise ValueError("Azure requires api_key, endpoint, and deployment/model name.")
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
            )
            self.model = model
        elif self.provider == "openai":
            if not all([api_key, model]):
                raise ValueError("OpenAI requires api_key and model name.")
            self.client = OpenAI(api_key=api_key)
            self.model = model
        else:
            raise ValueError("Invalid provider. Choose 'openai' or 'azure'.")

    def run(self, prompt: str, max_tokens: int = 100) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
