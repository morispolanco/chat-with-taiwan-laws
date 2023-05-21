import os
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Variables de entorno
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

# Cargar embeddings y base de datos
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings)

llm = ChatOpenAI(model_name=OPENAI_MODEL, temperature=0)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer"),
    retriever=db.as_retriever(),
    max_tokens_limit=3500,
)

st.set_page_config(
    page_title="Chat con las leyes de Taiwán",
    page_icon=":robot:"
)

st.title("台灣法規 Chat AI")
st.markdown("""
[![](https://img.shields.io/badge/tpai/chat_with_taiwan_laws-grey?style=flat-square&logo=github)](https://github.com/tpai/chat-with-taiwan-laws)
""")
st.markdown("""
Esta herramienta utiliza la [Base de Datos Nacional de Leyes de Taiwán](https://law.moj.gov.tw/Hot/AddHotLaw.ashx?pcode=B0000001), que incluye las leyes civiles de Taiwán, el Código Penal de la República de China, la Ley de Procedimiento Penal, la Ley de Normas Laborales, el Reglamento de Jubilación de Empleados, y el Reglamento de Seguridad y Salud Ocupacional. Estos archivos son en formato PDF. Esta herramienta es solo para fines de investigación y aprendizaje. Si necesita asesoramiento legal, consulte a un abogado profesional.
""")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'memory' not in st.session_state:
    st.session_state['memory'] = ''

def get_text():
    input_text = st.text_input("Ingrese la conversación:", "Hola", key="input")
    return input_text 

question = get_text()

if question:
    with st.spinner("🤖 Generando respuesta, por favor espera..."):
        humanMessage = question
        output = chain({"question": f"Historial de conversación:\n{st.session_state['memory']}\n---\n{humanMessage} Por favor, responda en chino tradicional de Taiwán de manera simple"})
        aiMessage = output["answer"]
        st.session_state['memory'] += f"Tú: {humanMessage}\nAI: {aiMessage}\n"
        st.session_state.past.append(question)
        st.session_state.generated.append(aiMessage)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
