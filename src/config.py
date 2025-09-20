import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
SOURCES_DIR = os.path.join(DATA_DIR, "sources")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")

# Ensure directories exist
for path in [SOURCES_DIR, PROCESSED_DIR, EMBEDDINGS_DIR]:
    os.makedirs(path, exist_ok=True)

# API keys and config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE", "openai")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2023-05-15")

# Set OpenAI client config
import openai
if OPENAI_API_TYPE == "openai":
    openai.api_key = OPENAI_API_KEY
    openai.api_type = "openai"
elif OPENAI_API_TYPE == "azure":
    openai.api_key = AZURE_OPENAI_API_KEY
    openai.api_type = "azure"
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_version = OPENAI_API_VERSION
    # Optionally set deployment name if needed elsewhere

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-ada-002")

os.environ["LITELLM_PROVIDER"] = os.getenv("OPENAI_API_TYPE")
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = os.getenv("OPENAI_API_VERSION")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Validar que as variáveis de ambiente necessárias estão presentes
if not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"):
    raise ValueError("As variáveis de ambiente de Azure OpenAI (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY) não estão configuradas.")

os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_DEPLOYMENT_NAME"] = os.getenv("AZURE_DEPLOYMENT_NAME")


for k in ["AZURE_API_KEY", "AZURE_API_BASE", "AZURE_API_VERSION", "AZURE_DEPLOYMENT_NAME"]:
    print(k, "=", os.getenv(k))