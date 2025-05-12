# augmentador_botanico.py
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from faiss_db import add_document_to_index
from qa_generator import generate_qa_from_text  # Assumindo que esse módulo existe

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.5, model="gpt-4o")

# Prompt para gramas
prompt_graminea = ChatPromptTemplate.from_template("""
Você é um botânico paisagista especializado em gramíneas. Reformule e expanda o texto abaixo com:
- Descrições mais técnicas (porte, raízes, propagação)
- Aplicações paisagísticas (ex: campo esportivo, cobertura de talude)
- Condições ideais de luz, solo, irrigação e poda
- Tolerância ao pisoteio e clima

Texto original:
{text}

Texto expandido:
""")

# Prompt para outras plantas
prompt_planta = ChatPromptTemplate.from_template("""
Você é um botânico paisagista. Expanda e reestruture a descrição da planta abaixo para torná-la mais completa, seguindo os critérios:

1. **Classificação**: nome científico (_itálico_), nome popular, origem e família.
2. **Características morfológicas**: porte, tipo de folhas, flores, frutos, sistema radicular.
3. **Adaptação ao ambiente**: luz solar (pleno, meia-sombra), tipo de solo, irrigação, clima.
4. **Aplicação paisagística**: onde e como usar essa planta em jardins ou espaços urbanos.
5. **Cuidados e manutenção**: podas, fertilização, pragas e doenças comuns.
6. **Conservação**: se aplicável, citar status de preservação e ameaças.

Evite repetir frases. Expanda termos técnicos. Não invente fatos.

Texto original:
{text}

Texto expandido:
""")


def expandir(texto, tipo):
    if tipo == "graminea":
        return (prompt_graminea | llm).invoke({"text": texto}).content
    elif tipo == "planta":
        return (prompt_planta | llm).invoke({"text": texto}).content
    else:
        return texto


def salvar_md_augmented(nome_arquivo, texto_exp):
    nome_base = nome_arquivo.rsplit(".md", 1)[0] + "_augmented.md"
    caminho = os.path.join("data", "md", nome_base)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto_exp)
    return nome_base, texto_exp


def detectar_tipo(texto):
    texto_lower = texto.lower()
    if any(p in texto_lower for p in ["grama", "gram\u00ednea", "zoysia", "esmeralda", "batatais", "bermuda"]):
        return "graminea"
    elif any(p in texto_lower for p in ["planta", "esp\u00e9cie", "arbusto", "folha", "flor", "herb\u00e1cea"]):
        return "planta"
    return None


def processar_documento_botanico(nome_arquivo, conteudo_md):
    tipo = detectar_tipo(conteudo_md)
    if tipo:
        texto_exp = expandir(conteudo_md, tipo)
        nome_salvo, texto_salvo = salvar_md_augmented(nome_arquivo, texto_exp)

        # Indexa no FAISS
        add_document_to_index(texto_salvo, nome_salvo)

        # Gera e indexa QAs
        qas = generate_qa_from_text(texto_salvo, nome_salvo)  # Você deve definir essa função em qa_generator.py
        if qas:
            add_document_to_index(qas, nome_salvo + "_qas")

        label = "gram\u00ednea" if tipo == "graminea" else "esp\u00e9cie vegetal"
        return f"{label.capitalize()} enriquecida, indexada e QAs gerados como {nome_salvo}"
    return "Documento n\u00e3o identificado como planta ou grama. Nenhuma altera\u00e7\u00e3o aplicada."


# Teste local
if __name__ == "__main__":
    arquivo_teste = "data/md/teste.md"
    with open(arquivo_teste, encoding="utf-8") as f:
        texto = f.read()

    print(processar_documento_botanico("teste.md", texto))