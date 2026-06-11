import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import time

# --- 1. CONFIGURACIÓN DE PÁGINA Y UI/UX IMPRESIONANTE ---
st.set_page_config(page_title="Simulador DO | Nivel 8M2", page_icon="🏢", layout="wide")

# Inyección de CSS (Diseño Neón Corporativo)
st.markdown("""
    <style>
    /* Fondo principal y textos */
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    /* Estilo de la cabecera principal */
    .big-font {
        font-size: 45px !important;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }
    /* Estilo de la barra lateral */
    [data-testid="stSidebar"] {
        background-color: #1A1C23;
        border-right: 1px solid #2D3139;
    }
    /* Tarjetas de métricas */
    [data-testid="stMetricValue"] {
        color: #00C9FF !important;
        font-size: 35px !important;
        font-weight: bold;
    }
    /* Botón de Reto */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #00C9FF 0%, #0072FF 100%);
        color: white;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(0, 201, 255, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. SISTEMA DE GAMIFICACIÓN (Cálculo de Niveles) ---
if "puntos_xp" not in st.session_state:
    st.session_state.puntos_xp = 0
if "mensajes_chat" not in st.session_state:
    st.session_state.mensajes_chat = [{"role": "assistant", "content": "Bienvenido al corporativo, Consultor Junior. ¿Qué caso de Desarrollo Organizacional analizaremos hoy?"}]

# Matemáticas de progreso: Cada 100 XP es un nivel nuevo
nivel_actual = (st.session_state.puntos_xp // 100) + 1
xp_en_nivel = st.session_state.puntos_xp % 100
progreso_porcentaje = xp_en_nivel / 100.0

# --- 3. BARRA LATERAL (Panel de Jugador) ---
with st.sidebar:
    st.markdown("### 🏆 Perfil del Consultor")
    st.metric(label=f"Nivel Corporativo", value=nivel_actual)
    
    st.markdown(f"**Progreso hacia el Nivel {nivel_actual + 1}**")
    st.progress(progreso_porcentaje)
    st.write(f"🌟 {st.session_state.puntos_xp} XP Totales")
    
    st.divider()
    st.markdown("#### 🎯 Objetivos de Hoy")
    st.info("1. Resolver dudas teóricas.\n2. Completar retos del Director.\n3. Acumular +10 XP por reto.")

# --- 4. CABECERA PRINCIPAL ---
st.markdown('<p class="big-font">Agencia de Desarrollo Organizacional</p>', unsafe_allow_html=True)
st.caption("🖥️ Plataforma de Simulación y Consultoría Académica | Powered by AI")
st.divider()

# --- 5. CONFIGURACIÓN DE SEGURIDAD E IA (Rigor Extremo) ---
api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ Enlace de seguridad roto: Falta tu llave de Google (GOOGLE_API_KEY) en la carpeta secreta.")
    st.stop()

# Temperatura 0 garantizada para rigor analítico
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=api_key)

# --- 6. PROCESAMIENTO DOCUMENTAL ---
@st.cache_resource(show_spinner=False)
def preparar_base_de_datos():
    loader = PyPDFDirectoryLoader("materiales")
    documentos = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    textos_cortados = text_splitter.split_documents(documentos)
    vectorstore = Chroma.from_documents(documents=textos_cortados, embedding=embeddings)
    return vectorstore

with st.spinner("Sincronizando manuales corporativos..."):
    vectorstore = preparar_base_de_datos()
    retriever = vectorstore.as_retriever()

# --- 7. INGENIERÍA DE PROMPTS (El Cerebro del Director) ---
prompt_template = """
Eres el Director Senior de una firma de Desarrollo Organizacional. El usuario es un Consultor Junior.
Tu objetivo es orientarlo basándote ÚNICAMENTE en el siguiente contexto.

Contexto:
{context}

Pregunta del Consultor: {question}

Instrucciones:
1. Responde con rigor académico usando solo el contexto. Si no está, di: "Esa metodología no está en nuestros manuales, revisa la planeación."
2. Tono: Profesional, implacable, pero constructivo.
3. RETO FINAL: Termina tu respuesta proponiendo un mini-caso práctico rápido sobre lo explicado para poner a prueba al consultor.

Respuesta del Director:
"""
PROMPT = PromptTemplate.from_template(prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# --- 8. INTERFAZ DE CHAT FLUIDA (UI/UX) ---
# Mostrar el historial de conversación
for mensaje in st.session_state.mensajes_chat:
    with st.chat_message(mensaje["role"], avatar="🏢" if mensaje["role"] == "assistant" else "👤"):
        st.markdown(mensaje["content"])

# Capturar la nueva pregunta del alumno
pregunta_alumno = st.chat_input("Escribe tu duda o análisis aquí, Consultor...")

if pregunta_alumno:
    # 1. Guardar y mostrar la pregunta del usuario
    st.session_state.mensajes_chat.append({"role": "user", "content": pregunta_alumno})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta_alumno)
    
  # 2. Generar y mostrar la respuesta del Director con Streaming (Ultrarrápido)
    with st.chat_message("assistant", avatar="🏢"):
        # Ejecuta la cadena en modo streaming y escribe los tokens en tiempo real
        respuesta = st.write_stream(qa_chain.stream(pregunta_alumno))
        
    # Guardar la respuesta completa en el historial
    st.session_state.mensajes_chat.append({"role": "assistant", "content": respuesta})
    
    # El Escudo Protector contra límites de cuota (Capa Gratuita)
    try:
        with st.chat_message("assistant", avatar="🏢"):
            respuesta = st.write_stream(qa_chain.stream(pregunta_alumno))
            st.session_state.mensajes_chat.append({"role": "assistant", "content": respuesta})
            
    except Exception as e:
        # Verifica si el error es por límite de peticiones (429)
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            st.warning("⏳ Límite de procesamiento corporativo alcanzado. Por favor, reflexiona sobre el caso y espera 1 minuto entero antes de enviar tu siguiente análisis.")
        else:
            st.warning("⚠️ El corporativo está experimentando intermitencias. Por favor, espera 30 segundos y vuelve a intentarlo.")

# --- 9. BOTÓN DE RECOMPENSA (Lógica de Juego) ---
# Se dibuja solo si hay al menos una interacción
if len(st.session_state.mensajes_chat) > 1:
    st.write("") # Espacio
    cols = st.columns([1, 2, 1])
    with cols[1]:
        if st.button("🚀 RESOLVÍ EL RETO DEL DIRECTOR (+10 XP)"):
            st.session_state.puntos_xp += 10
            st.toast('¡Excelente análisis! Has ganado +10 XP', icon='✅')
            st.balloons()
            time.sleep(1.5) # Pausa dramática antes de recargar
            st.rerun()
