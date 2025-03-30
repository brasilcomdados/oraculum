import streamlit as st
import os
import shutil
import numpy as np
import pandas as pd
from faiss_db import get_embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "data/faiss_index")


def delete_from_faiss(filename: str):
    """Remove documentos do índice FAISS (mantido original)"""
    try:
        index = FAISS.load_local(
            INDEX_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )

        keep_docs = []
        for doc in index.docstore._dict.values():
            if isinstance(doc.metadata, dict) and doc.metadata.get('source') != filename:
                keep_docs.append({
                    "text": doc.page_content,
                    "meta": doc.metadata
                })

        if keep_docs:
            new_index = FAISS.from_embeddings(
                text_embeddings=[
                    (doc["text"], get_embeddings().embed_query(doc["text"]))
                    for doc in keep_docs
                ],
                embedding=get_embeddings(),
                metadatas=[doc["meta"] for doc in keep_docs]
            )
            new_index.save_local(INDEX_DIR)
        else:
            shutil.rmtree(INDEX_DIR)

        return True

    except Exception as e:
        st.error(f"Erro na exclusão: {str(e)}")
        return False


def get_vector_data(index):
    """Extrai todos os chunks vetoriais individualmente"""
    vector_data = []
    for i in range(index.index.ntotal):
        try:
            doc = index.docstore.search(index.index_to_docstore_id[i])
            vector = index.index.reconstruct(i)

            vector_data.append({
                "ID Chunk": i + 1,
                "Fonte": doc.metadata.get("source", "Desconhecido"),
                "Conteúdo": doc.page_content[:150] + "...",
                "Metadados": str(doc.metadata),
                "Dimensões": len(vector),
                "Vetor (5 primeiras)": f"[{', '.join(f'{v:.4f}' for v in vector[:5])}...]",
                "Vetor Completo": vector.tolist()  # Para exportação
            })
        except Exception as e:
            st.error(f"Erro ao processar chunk {i}: {str(e)}")
    return pd.DataFrame(vector_data)


def show_vector_table(index):
    """Exibe tabela detalhada com todos os chunks"""
    st.write("### Registros Vetoriais - Visão Detalhada por Chunk")

    df = get_vector_data(index)

    # Configuração de paginação
    col1, col2 = st.columns([2, 3])
    with col1:
        page_size = st.selectbox("Registros por página", [5, 10, 25, 100], index=0)
    with col2:
        page_number = st.number_input("Página",
                                      min_value=1,
                                      max_value=(len(df) // page_size) + 1,
                                      value=1)

    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size

    # Exibe tabela principal
    st.dataframe(
        df.iloc[start_idx:end_idx][["ID Chunk", "Fonte", "Conteúdo", "Metadados", "Dimensões", "Vetor (5 primeiras)"]],
        use_container_width=True,
        height=600,
        hide_index=True,
        column_config={
            "Vetor (5 primeiras)": st.column_config.Column(
                width="medium",
                help="Primeiras 5 dimensões do vetor"
            )
        }
    )

    # Controles de exportação
    with st.expander("🔧 Opções de Exportação"):
        st.download_button(
            "📥 Exportar dados completos (CSV)",
            df.to_csv(index=False).encode(),
            "vetores_completos.csv",
            "text/csv",
            help="Inclui vetores completos e todos os metadados"
        )

        st.download_button(
            "📥 Exportar vetores brutos (JSON)",
            df[["ID Chunk", "Vetor Completo"]].to_json(indent=2),
            "vetores_brutos.json",
            "application/json"
        )

    # Exibição do vetor completo selecionado
    selected_id = st.number_input("Insira o ID do chunk para ver o vetor completo:",
                                  min_value=1,
                                  max_value=len(df),
                                  value=1)

    if selected_id:
        full_vector = df[df["ID Chunk"] == selected_id]["Vetor Completo"].iloc[0]
        with st.expander(f"🔎 Vetor Completo - Chunk {selected_id}"):
            st.write(f"Dimensões: {len(full_vector)}")
            st.code(f"[{', '.join(f'{v:.6f}' for v in full_vector)}]")

def show_faiss_manager():
    """Interface principal com abas"""
    try:
        if not os.path.exists(INDEX_DIR):
            st.warning("Nenhum índice FAISS encontrado!")
            return

        index = FAISS.load_local(
            INDEX_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True
        )

        tab1, tab2 = st.tabs(["Gestão do Banco", "Visualização dos Vetores"])

        with tab1:
            st.subheader("🗃️ Gestão de Documentos")
            sources = {}
            for doc in index.docstore._dict.values():
                if isinstance(doc.metadata, dict):
                    source = doc.metadata.get('source', 'Desconhecido')
                    sources[source] = sources.get(source, 0) + 1

            for source, count in sources.items():
                with st.expander(f"{source} ({count} registros)"):
                    if st.button(f"Excluir todos de {source}", key=f"del_{source}"):
                        if delete_from_faiss(source):
                            st.rerun()

        with tab2:
            st.subheader("🔍 Visualização Vetorizada")
            show_vector_table(index)

    except Exception as e:
        st.error(f"Erro ao carregar índice: {str(e)}")
