"""
faiss_db.py
-----------
Módulo responsável por gerenciar o índice FAISS para persistência e consulta vetorizada de documentos.
Utiliza as bibliotecas: faiss-cpu, langchain e openai.
"""

import os
from dotenv import load_dotenv
from langchain_openai import (
    OpenAIEmbeddings,
)
from langchain_community.vectorstores import FAISS
import streamlit as st  # Adição importante para acessar secrets
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Diretório onde o índice FAISS será salvo
INDEX_DIR = os.getenv("INDEX_DIR", "data/faiss_index")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING", "text-embedding-3-small")
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", 1000))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", 200))


def get_embeddings():
    """Factory method para criação segura dos embeddings com API Key"""
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=MODEL_EMBEDDING
    )


def init_faiss_index():
    """Inicializa o índice FAISS se não existir"""
    os.makedirs(INDEX_DIR, exist_ok=True)

    if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        empty_index = FAISS.from_texts(
            texts=[""],
            embedding=get_embeddings(),
            metadatas=[{"source": "system_init"}]
        )
        empty_index.save_local(INDEX_DIR)
        st.toast("Índice FAISS inicializado com sucesso")


def add_document_to_index(document_text: str, filename: str,
                          chunk_size: int = DEFAULT_CHUNK_SIZE,
                          chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
    """
    Adiciona documento com chunking controlado
    """
    # Cria o splitter com parâmetros configuráveis
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    )

    # Divide o documento em chunks
    chunks = text_splitter.split_text(document_text)

    # Prepara metadados para cada chunk
    metadatas = [{"source": filename} for _ in chunks]

    embeddings = get_embeddings()

    if os.path.exists(INDEX_DIR):
        index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        index.add_texts(
            texts=chunks,
            metadatas=metadatas
        )
    else:
        index = FAISS.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=metadatas
        )

    index.save_local(INDEX_DIR)
    return index

def search_documents(query: str, k: int = 4):
    """
    Busca retornando documentos com metadados
    """
    try:
        if not os.path.exists(INDEX_DIR):
            return []

        embeddings = get_embeddings()
        index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return index.similarity_search_with_score(query, k=k)

    except Exception as e:
        st.error(f"Erro na busca FAISS: {str(e)}")
        return []


def list_faiss_documents():
    """Lista todos os documentos únicos no índice FAISS"""
    if not os.path.exists(INDEX_DIR):
        return []

    embeddings = get_embeddings()
    index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    return list({
        meta['source']
        for _, (_, meta) in index.docstore._dict.items()
        if 'source' in meta
    })
