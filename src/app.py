import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
import unicodedata

def clean_text(text):
    # Normalize the text to NFKD form and filter out non-ASCII characters
    normalized_text = unicodedata.normalize('NFKD', text)
    ascii_text = ''.join([c for c in normalized_text if ord(c) < 128])
    return ascii_text

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="101 LLM - Livro")
st.title("LLM - Livro")

llm = ChatOllama(
            model="llama3"
        )

def get_response(context, query, chat_history):
    template = """
    Você é um modelo de inteligencia artificial e só deve responder em português sobre o contexto definido abaixo

    Contexto: {context}

    Historico de conversa: {chat_history}

    Pergunta do usuário: {query}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "context": context,
        "chat_history": chat_history,
        "query": query
    })

uploaded_file = st.file_uploader("Escolha um arquivo em PDF")
# Carregar arquivo
if uploaded_file is not None:
    with open('files/'+uploaded_file.name, mode='wb') as w:
        w.write(uploaded_file.getvalue())
        
    loader = PyPDFLoader('files/'+uploaded_file.name)
    pages = loader.load_and_split()

    # Limpar o texto de cada página
    for page in pages:
        page.page_content = clean_text(page.page_content)

    # Vetorizar arquivo
    faiss_index = FAISS.from_documents(pages, OllamaEmbeddings(model="nomic-embed-text"))

    # Recuperar mensagens
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("IA"):
                st.markdown(message.content)

    # Enviar mensagens e salvar historico

    user_query = st.chat_input('Digite aqui sua pergunta.')
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(user_query))

        with st.chat_message("Human"): 
            st.markdown(user_query)

        context = [doc.page_content for doc in faiss_index.similarity_search(user_query, k=3)]
        ai_response = st.write_stream(get_response(context, user_query, st.session_state.chat_history))

        with st.chat_message("IA"):
            st.markdown(ai_response)

        st.session_state.chat_history.append(SystemMessage(ai_response))