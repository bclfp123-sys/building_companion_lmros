import pickle
import numpy as np
from src.embeddings.vector_store import VectorStore
from src.embeddings.embedder import get_embedding
from src.embeddings.embedder import generate_answer_openai

class RAGCourseAgent:
    def ai_rerank(self, query, candidates, use_summary=True, max_tokens=300):
        """
        Use LLM to select the most relevant chunks/summaries for the query.
        """
        context_list = []
        for i, (meta_item, score) in enumerate(candidates):
            chunk_content = meta_item.get("summary") if use_summary and meta_item.get("summary") else meta_item.get("text")
            context_list.append(f"Trecho {i+1}:\n{chunk_content}")

        context_str = "\n\n".join(context_list)
        prompt = (
            f"Pergunta do utilizador: {query}\n\n"
            f"Segue-se uma lista de excertos de documentos. Indica quais os trechos mais relevantes para responder à pergunta, justificando brevemente a escolha:\n\n"
            f"{context_str}\n\n"
            "Responde com os números dos trechos mais relevantes e uma breve justificação."
        )

        response = generate_answer_openai(prompt, model=self.openai_model, max_tokens=max_tokens)
        return response
    def __init__(self, threshold=0.4, openai_model="gpt-3.5-turbo"):
        # Load FAISS + meta
        self.vs = VectorStore()
        self.vs.load()  # loads faiss.index
        with open("embeddings/meta.pkl", "rb") as f:
            self.meta = pickle.load(f)
        self.openai_model = openai_model
        self.threshold = threshold  # minimum similarity to consider relevant

    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def retrieve(self, query: str, top_k=5, alpha=0.5, ai_rerank=True):
        """
        Recupera excertos relevantes usando embeddings do chunk e do resumo.
        alpha: peso para combinar as similaridades (0.5 = média simples)
        """
        # Embed the query for both chunk and summary
        query_embedding = get_embedding(query)
        query_summary_embedding = get_embedding(query)  # Optionally, use a different prompt for summary

        scores = []
        for meta_item in self.meta:
            chunk_emb = meta_item.get("chunk_embedding")
            summary_emb = meta_item.get("summary_embedding")
            if chunk_emb is not None and summary_emb is not None:
                chunk_score = self.cosine_similarity(query_embedding, chunk_emb)
                summary_score = self.cosine_similarity(query_summary_embedding, summary_emb)
                combined_score = alpha * chunk_score + (1 - alpha) * summary_score
            elif chunk_emb is not None:
                combined_score = self.cosine_similarity(query_embedding, chunk_emb)
            else:
                combined_score = 0
            scores.append((meta_item, combined_score))

        # Sort by combined score and keep more candidates for AI rerank
        scores = [item for item in scores if item[1] >= self.threshold]
        scores.sort(key=lambda x: x[1], reverse=True)
        candidates = scores[:max(20, top_k)]

        if ai_rerank and candidates:
            ai_response = self.ai_rerank(query, candidates)
            print("AI Rerank Response:", ai_response)
            # Optionally, parse ai_response to select the best chunks
            # For now, just return the top_k candidates
            return candidates[:top_k]
        else:
            return candidates[:top_k]

    def answer(self, query: str, top_k=5, max_tokens=500, use_summary=True):
        """
        Gerar uma resposta à pergunta usando o contexto recuperado.
        Se use_summary=True, utiliza os resumos dos excertos; caso contrário, utiliza o texto completo.
        """
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query, top_k=top_k)

        if not retrieved_chunks:
            return "❌ Desculpe, não tenho informação suficiente no corpus para responder a isso."

        # Passo 2: Preparar contexto da meta
        context_texts = []
        for chunk, score in retrieved_chunks:
            # Usa o resumo se disponível e solicitado, senão usa o texto completo
            chunk_content = chunk.get("summary") if use_summary and chunk.get("summary") else chunk.get("text")
            header = f"[Fonte: {chunk.get('file', 'desconhecido')} | Chunk: {chunk.get('chunk_id', 'N/D')}]"
            context_texts.append(f"{header}\n{chunk_content}")

        context = "\n\n".join(context_texts)
        print("------------------------------HERE------------------------------")
        print(context)
        asdasd

        # Passo 3: Construir prompt para LLM
        prompt = (
            f"És um assistente de ensino.\n\n"
            f"Utiliza o seguinte contexto extraído dos documentos para responder à pergunta do utilizador:\n\n"
            f"{context}\n\n"
            f"Pergunta do Utilizador: {query}\n\n"
            "Responde de forma clara e concisa usando apenas o contexto fornecido. "
            "Se o contexto não contiver a resposta, diz que não sabes."
        )

        # Step 4: Gerar resposta usando OpenAI
        return generate_answer_openai(prompt, model=self.openai_model, max_tokens=max_tokens)
