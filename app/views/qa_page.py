import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os
import re
import time
from collections import OrderedDict
from threading import Lock
from file_md import list_documents, get_document

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_QA_GENERATOR = os.getenv("MODEL_QA_GENERATOR")

# Configurações otimizadas
INITIAL_CHUNK_SIZE = 15000
MAX_WORKERS = 4


def dynamic_chunk_size(text_length):
    """Ajusta dinamicamente o tamanho dos chunks"""
    if text_length > 200000:  # Acima de 200k caracteres
        return 30000
    elif text_length > 100000:
        return 20000
    return INITIAL_CHUNK_SIZE


def chunk_document(text):
    """Divide o documento de forma otimizada"""
    chunk_size = dynamic_chunk_size(len(text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1),  # 10% de overlap
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    )
    return splitter.split_text(text)


def process_chunk(args):
    """Processa cada parte do documento"""
    chunk, prompt_template, params = args
    try:
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=params['temperature'],
            model=MODEL_QA_GENERATOR,
            max_retries=2,
            request_timeout=30
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm

        result = chain.invoke({
            "num_questions": params['questions_per_chunk'],
            "context_keywords": params['context_keywords'],
            "difficulty": params['difficulty'],
            "document_text": chunk
        }).content

        return result, None  # Resultado, Erro

    except Exception as e:
        return None, str(e)


def generate_qa_streaming(doc_text, prompt_text, params):
    """Processamento paralelo com exibição progressiva"""
    chunks = chunk_document(doc_text)
    total_chunks = len(chunks)

    if total_chunks == 0:
        return ""

    # Controle de estado
    st.session_state.start_time = time.time()
    st.session_state.qa_buffer = []
    st.session_state.processing_errors = []
    st.session_state.lock = Lock()
    st.session_state.completed_chunks = 0

    # Cálculo dinâmico de questões por chunk
    params['questions_per_chunk'] = max(1, params['num_questions'] // max(total_chunks, 1))

    # Configuração da interface
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_chunk, (chunk, prompt_text, params)): i
                   for i, chunk in enumerate(chunks)}

        for future in as_completed(futures):
            chunk_index = futures[future]
            result, error = future.result()

            with st.session_state.lock:
                st.session_state.completed_chunks += 1
                progress = st.session_state.completed_chunks / total_chunks
                progress_bar.progress(progress)

                elapsed = time.time() - st.session_state.start_time
                status_text.markdown(f"""
                    **Progresso:** {st.session_state.completed_chunks}/{total_chunks} chunks  
                    **Tempo decorrido:** {elapsed:.1f}s  
                    **QAs gerados:** {len(st.session_state.qa_buffer)}
                """)

                if result:
                    st.session_state.qa_buffer.append(result)
                    # Atualização parcial da interface
                    with results_container:
                        display_qa_chunk(result)

                if error:
                    st.session_state.processing_errors.append(error)

    # Processamento final
    final_content = clean_qa_content("\n\n".join(st.session_state.qa_buffer))

    # Limpeza do estado
    progress_bar.empty()
    status_text.empty()

    return final_content


def display_qa_chunk(content):
    """Exibe um conjunto parcial de QAs"""
    qa_pattern = r"\*\*Pergunta \d+:\*\*.*?(?=\n\*\*Pergunta \d+:\*\*|\Z)"
    qa_pairs = re.findall(qa_pattern, content, re.DOTALL)

    for pair in qa_pairs:
        with st.expander(f"⚡ QA Gerada", expanded=False):
            st.markdown(pair)


def clean_qa_content(content):
    """Otimização na limpeza de conteúdo"""
    qa_pairs = content.split("\n\n")
    seen = set()
    unique_pairs = []

    for pair in qa_pairs:
        simplified = re.sub(r'\s+', ' ', pair).strip()
        if simplified not in seen:
            seen.add(simplified)
            unique_pairs.append(pair)

    return "\n\n".join(unique_pairs)


def show_qa_generator():
    st.title("📝 Gerador de Perguntas e Respostas (Otimizado)")

    # Gerenciamento de estado
    session_defaults = {
        'qa_content': None,
        'show_results': False,
        'selected_doc': None,
        'doc_text': "",
        'processing': False
    }

    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Seletor de documentos
    docs = list_documents()
    if not docs:
        st.warning("Nenhum documento disponível. Faça upload primeiro.")
        return

    new_selection = st.selectbox(
        "Selecione o documento:",
        docs,
        index=0,
        key="selected_doc",
        on_change=lambda: st.session_state.update({
            'doc_text': get_document(st.session_state.selected_doc),
            'show_results': False
        })
    )

    # Atualização do documento
    if st.session_state.selected_doc != new_selection:
        st.session_state.selected_doc = new_selection
        st.session_state.doc_text = get_document(new_selection)
        st.rerun()

    # Estatísticas do documento
    if st.session_state.doc_text:
        with st.expander("📊 Métricas do Documento", expanded=True):
            col1, col2 = st.columns(2)
            text_len = len(st.session_state.doc_text)
            words = st.session_state.doc_text.split()

            with col1:
                st.metric("Caracteres", f"{text_len:,d}".replace(",", "."))
                st.metric("Palavras", f"{len(words):,d}".replace(",", "."))

            with col2:
                st.metric("Palavras Únicas", f"{len(set(words)):,d}".replace(",", "."))
                st.metric("Tamanho Médio Palavra", f"{sum(len(w) for w in words) / len(words):.1f}")

    # Formulário de geração
    with st.form("qa_form"):
        default_prompt = """Você é um especialista em criação de conteúdos educacionais. 
             Gere no mínimo de {num_questions} perguntas e respostas baseadas no documento abaixo, seguindo estas regras:

             1. Foco nos contextos: {context_keywords} (priorizar estes termos)
             2. Formato de resposta: **Pergunta X:** [texto] \\n\\n **Resposta X:** [texto]
             3. Nível de detalhe: adequado para profissionais de nível {difficulty}
             4. Inclua exemplos práticos quando relevante

             Documento:
             {document_text}"""

        prompt_text = st.text_area(
            "Instruções para geração:",
            value=default_prompt,
            height=200
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            num_questions = st.number_input("Número de QAs", 1, 500, 50)
        with col2:
            difficulty = st.selectbox("Dificuldade", ["Iniciante", "Intermediário", "Avançado"])
        with col3:
            temperature = st.slider("Criatividade", 0.0, 1.0, 0.5)

        context_keywords = st.text_input("Palavras-chave (separadas por vírgula)")

        if st.form_submit_button("🚀 Gerar QAs") and st.session_state.doc_text:
            st.session_state.processing = True
            st.session_state.show_results = False

            params = {
                'num_questions': num_questions,
                'context_keywords': context_keywords,
                'difficulty': difficulty,
                'temperature': temperature
            }

            try:
                st.session_state.qa_content = generate_qa_streaming(
                    st.session_state.doc_text,
                    prompt_text,
                    params
                )
                st.session_state.show_results = True

                # Exibir métricas finais
                elapsed = time.time() - st.session_state.start_time
                st.toast(f"Processo concluído em {elapsed:.1f} segundos", icon="✅")

            except Exception as e:
                st.error(f"Erro crítico: {str(e)}")
            finally:
                st.session_state.processing = False

    # Exibição de resultados
    if st.session_state.show_results and st.session_state.qa_content:
        st.markdown("---")
        st.subheader("Resultado Final")

        with st.expander("📋 Visualizar Todas as QAs", expanded=True):
            display_qa_results(st.session_state.qa_content)

        st.download_button(
            "💾 Baixar QAs",
            st.session_state.qa_content,
            file_name="qas_gerados.md",
            mime="text/markdown"
        )


def display_qa_results(content):
    """Exibição otimizada de resultados finais"""
    qa_pairs = content.split("\n\n")

    for i, pair in enumerate(qa_pairs, 1):
        with st.container():
            st.markdown(f"**Pergunta {i}**")
            st.markdown(pair.replace("\\n\\n", "\n\n"))
            st.write("---")


if __name__ == "__main__":
    show_qa_generator()