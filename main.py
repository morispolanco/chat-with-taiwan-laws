import os
import streamlit as st

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Variables de entorno
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

# Cargar embeddings y base de datos
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_index", embeddings)

# Crear una cadena de recuperación con el modelo ChatOpenAI
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=ChatOpenAI(model_name=OPENAI_MODEL, temperature=0), chain_type="stuff", retriever=db.as_retriever(), max_tokens_limit=3500, reduce_k_below_max_tokens=True)

# Campo de entrada y área de texto de Streamlit
st.title("Preguntas y respuestas sobre leyes de Taiwán")
st.markdown("""
[![](https://img.shields.io/badge/tpai/chat_with_taiwan_laws-grey?style=flat-square&logo=github)](https://github.com/tpai/chat-with-taiwan-laws)
""")
# Bloque de descripción
st.markdown("""
Esta herramienta utiliza la [Base de Datos Nacional de Leyes de Taiwán](https://law.moj.gov.tw/Hot/AddHotLaw.ashx?pcode=B0000001) que incluye las leyes civiles, la ley de procedimiento civil, la ley de ejecución del procedimiento civil, el código penal de la República de China, la ley de ejecución del código penal, la ley de procedimiento penal, la ley de ejecución del procedimiento penal, la ley de mantenimiento del orden social, el reglamento de tratamiento de casos de violación de la ley de mantenimiento del orden social, el reglamento de contacto entre tribunales locales y agencias de policía en el tratamiento de casos de violación de la ley de mantenimiento del orden social, la ley de prevención del acoso y hostigamiento, el reglamento de ejecución de la ley de prevención del acoso y hostigamiento, y el reglamento de ejecución de la orden de protección en casos de acoso y hostigamiento. Estos archivos son en formato PDF. Esta herramienta es solo para fines de investigación y aprendizaje. Si necesita asesoramiento legal, consulte a un abogado profesional.
""")
question = st.text_input("Por favor, ingrese su pregunta:")
if question:
    with st.spinner("🤖 Pensando, por favor espera..."):
        output = chain({"question": f"{question} Por favor, responda en chino tradicional de Taiwán"}, return_only_outputs=True)
    st.text_area("🤖:", value=output["answer"], height=200)
