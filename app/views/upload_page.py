import streamlit as st
from file_md import persist_document, list_documents, get_document, update_document
from utils import convert_file_to_md
from faiss_db import add_document_to_index
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#sugestão de melhoria: modificar o modelo para gpt-4o inserindo menos ruído com a necesidade de luz 
def format_with_ai(text: str) -> str:
    """
    Melhora a formatação de documentos Markdown mantendo a integridade das informações
    """
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um paisagista especializado em palmeiras e gramíneas. Reformate descrições botânicas seguindo esta estrutura:

            1. **Classificação Botânica**: nome científico (_itálico_), nome popular, origem, família
            2. **Necessidade de Luz**: sol pleno, meia sombra, sombra; relacione com horas de luz direta/indireta
            3. **Distribuição Natural**: localização nativa e ecossistema
            4. **Descrição Morfológica**: altura, tronco, folhas, flores, frutos (com destaque **negrito** para características)
            5. **Cultivo**: tolerância climática, tipo de solo, exigência hídrica
            6. **Aplicações Paisagísticas**: sugestões de uso em projetos
            7. **Cuidado e Manutenção**: podas, irrigação, pragas
            8. **Conservação**: status IUCN, ameaças

            Use cabeçalhos hierárquicos (#, ##) e destaque termos técnicos. Nunca invente informações.
            """),
            ("human", "Texto original:\n{text}\n\nTexto formatado:")
        ])

        chain = prompt | ChatOpenAI(
            api_key=OPENAI_API_KEY,
            temperature=0.2,  # Balanceia criatividade e consistência
            model="gpt-4o"
        )

        response = chain.invoke({"text": text})
        return response.content

    except Exception as e:
        st.error(f"Erro na formatação: {str(e)}")
        return text


def show():
    st.title("Upload e Processamento de Arquivo")
    st.write(
        "Área destinada ao upload de arquivos, extração de informações com o docling e persistência em banco vetorizado.")

    # Sidebar para upload
    st.sidebar.header("Opções de Upload")
    global uploaded_file
    uploaded_file = st.sidebar.file_uploader("Selecione um arquivo", type=["pdf", "docx", "txt", "md"])

    if uploaded_file:
        if st.sidebar.button("Processar arquivo"):
            with st.spinner("Processando arquivo..."):
                try:
                    global md_content
                    md_content = convert_file_to_md(uploaded_file)
                    persist_document(uploaded_file.name, md_content)
                    st.session_state.last_uploaded = f"{uploaded_file.name.rsplit('.', 1)[0]}.md"
                    st.success("Arquivo processado e salvo com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao processar o arquivo: {e}")

    st.subheader("Arquivos Processados")
    docs = list_documents()

    if docs:
        default_index = 0
        if "last_uploaded" in st.session_state and st.session_state.last_uploaded in docs:
            default_index = docs.index(st.session_state.last_uploaded)

        selected_doc = st.selectbox("Selecione um arquivo para visualizar", docs, index=default_index)

        if "edit_mode" not in st.session_state:
            st.session_state.edit_mode = False

        # Seção de persistência modificada
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Persistir no FAISS", help="Armazena o documento no banco vetorial"):
                with st.spinner("Persistindo documento no FAISS..."):
                    try:
                        doc_text = get_document(selected_doc)
                        # Passe o nome do arquivo como segundo parâmetro
                        add_document_to_index(doc_text, selected_doc)  # selected_doc já contém o nome
                        st.success("Documento persistido no FAISS com sucesso!")
                    except Exception as e:
                        st.error(f"Erro ao persistir documento no FAISS: {e}")

        with col2:
            if st.button("Editar" if not st.session_state.edit_mode else "Sair do modo edição"):
                st.session_state.edit_mode = not st.session_state.edit_mode

        if st.session_state.edit_mode:
            md_content = get_document(selected_doc)
            col1, col2 = st.columns([5, 1])
            with col1:
                edited_content = st.text_area(
                    "Editar Conteúdo",
                    value=md_content,
                    height=400,
                    help="Modifique o conteúdo Markdown diretamente ou use o botão de formatação automática"
                )

            with col2:
                st.markdown("**Ações:**")
                if st.button(
                        "🪄 Auto-Formatar",
                        help="Melhora a formatação mantendo o conteúdo original"
                ):
                    edited_content = format_with_ai(edited_content)
                    st.session_state.last_edited = edited_content
                    st.rerun()

                if st.button(
                        "📤 Salvar",
                        type="primary",
                        help="Persiste as alterações no documento"
                ):
                    update_document(selected_doc, edited_content)
                    st.session_state.edit_mode = False
                    st.success("Documento atualizado!")
                    st.rerun()

        if st.session_state.edit_mode:
            md_content = get_document(selected_doc)
            edited_content = st.text_area("Edite o conteúdo Markdown", value=md_content, height=300)
            if st.button("Salvar Alterações"):
                update_document(selected_doc, edited_content)
                st.session_state.edit_mode = False
                st.success("Conteúdo atualizado!")
                st.rerun()
        else:
            st.markdown(get_document(selected_doc))

    else:
        st.info("Nenhum arquivo processado ainda.")

from augmentador_botanico import processar_documento_botanico

...

msg = processar_documento_botanico(uploaded_file.name, md_content)
if "enriquecida" in msg:
    st.toast(msg, icon="🌿")