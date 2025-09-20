import os
from src.ingest.extractor import extract_file, extract_pdf
from src.ingest.chunker import chunk_text
from src.embeddings.embedder import batch_embed
from src.embeddings.vector_store import VectorStore
from src.utils.logger import log_extraction_report
from src.agents.rag_agent import RAGCourseAgent
from src.config import SOURCES_DIR
from src.agents.summary_agent import SummaryAgent
from src.agents.llm_client import LLMClient


def run_pipeline():
    vs = VectorStore()
    
    for filename in os.listdir(SOURCES_DIR):
        file_path = os.path.join(SOURCES_DIR, filename)
    print(f"A processar {file_path} ...")

    # --- Step 1: extract ---
    if filename.lower().endswith(".pdf"):
        data = extract_pdf(file_path)  # returns dict
        text = data["text"]
        tables = data["tables"]
    else:
        text = extract_file(file_path)  # plain string
        tables = []

    # --- Step 2: chunk ---
    chunks = chunk_text(text)

    summarizer = SummaryAgent()

    # --- Step 3: log QA report ---
    #log_extraction_report(filename, text, chunks)

    # --- Step 4: embeddings ---
    embeddings = batch_embed(chunks)

    # --- Step 5: store in vector DB ---
    metadata = [{"file": filename, "chunk_id": i, "text": chunk,"summary": summarizer.summarize(chunk)}
                for i, chunk in enumerate(chunks)]
    
    
    vs.add(embeddings, metadata)

    # --- Step 6: optional table save ---
    if tables:
        for i, table in enumerate(tables):
            table_path = os.path.join("data/processed", f"{filename}_table_{i}.csv")
            with open(table_path, "w", encoding="utf-8") as f:
                for row in table:
                    f.write(",".join([cell if cell else "" for cell in row]) + "\n")

    vs.save()
    print("Pipeline concluído. Índice FAISS guardado.")



def run_rag_agent(user_input = "Faz uma pergunta"):
    print("🔎 A inicializar o Agente de Curso RAG...")
    agent = RAGCourseAgent()

    while True:
        query = input(user_input)
        if query.lower() in ["sair", "terminar"]:
            break

        answer = agent.answer(query)
        print("\n📘 Resposta:\n", answer)

if __name__ == "__main__":
    #run_pipeline()
    
    user_input = "Posso construir uma varanda no segundo andar da minha casa, voltada para a rua, e quais são as restrições de dimensão ou distância ao passeio segundo o regulamento das edificações urbanas em Portugal?"
    run_rag_agent(user_input)


