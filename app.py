import streamlit as st
import os
import sys
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Cargar variables de entorno
load_dotenv()

# Asegurar que el path del proyecto esté en el sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import rag_engine

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="BimBam Buy - Copilot RAG IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para una estética corporativa Premium y Moderna (Glassmorphism & Gradients)
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Aplicar fuente Outfit */
html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif;
    background-color: #0b0f19;
    color: #f3f4f6;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #0d1321;
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] .stMarkdown h1, 
section[data-testid="stSidebar"] .stMarkdown h2, 
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8;
    font-weight: 600;
}

/* Gradient Header */
.main-header {
    background: linear-gradient(135deg, #0284c7 0%, #7c3aed 100%);
    padding: 2.5rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.3);
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    color: #ffffff;
    letter-spacing: -0.025em;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}

.main-header p {
    font-size: 1.1rem;
    margin-top: 0.5rem;
    margin-bottom: 0;
    color: #e2e8f0;
    font-weight: 300;
}

/* Glassmorphism Cards for Suggestion Chips */
.suggestion-card {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    height: 100%;
}

.suggestion-card:hover {
    background: rgba(30, 41, 59, 0.8);
    border-color: #38bdf8;
    transform: translateY(-4px);
    box-shadow: 0 10px 20px -10px rgba(56, 189, 248, 0.4);
}

.suggestion-card h4 {
    margin: 0;
    color: #38bdf8;
    font-size: 1rem;
    font-weight: 600;
}

.suggestion-card p {
    margin: 0.5rem 0 0 0;
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.4;
}

/* Chat bubble styling overrides */
div[data-testid="stChatMessage"] {
    background-color: rgba(30, 41, 59, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

/* User Message specific gradient border or background */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    border-left: 4px solid #7c3aed;
    background-color: rgba(124, 58, 237, 0.05);
}

/* AI Message specific border */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    border-left: 4px solid #38bdf8;
    background-color: rgba(56, 189, 248, 0.05);
}

/* Status Indicator */
.status-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
    background-color: rgba(16, 185, 129, 0.1);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.2);
}

/* Source Box Styling */
.source-box {
    background-color: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 0.8rem;
    margin-top: 0.5rem;
    font-size: 0.85rem;
}

.source-title {
    color: #f43f5e;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 0.3rem;
}

.source-content {
    color: #94a3b8;
    font-style: italic;
    margin-top: 0.2rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/gradient/100/artificial-intelligence.png", width=70)
    st.markdown("## BimBam Buy")
    st.markdown("*Copilot IA de Soporte Interno*")
    st.write("---")
    
    st.markdown("### 📋 Documentos Indexados")
    docs_list = [
        "Guía de tiempos y Costos de Envío",
        "Manual de Garantía de Productos",
        "Política de Reembolsos y Devoluciones",
        "Preguntas Frecuentes sobre Métodos de Pago",
        "Programa de Afiliados"
    ]
    for d in docs_list:
        st.markdown(f"- 📄 {d}")
        
    st.write("---")
    
    st.markdown("### ⚙️ Motor RAG")
    st.markdown("**LLM:** Groq (`llama-3.1-8b-instant`)")
    st.markdown("**Embeddings:** Google (`gemini-embedding-001`)")
    st.markdown("**Base de Datos:** FAISS (Local)")
    
    st.write("---")
    
    # Botón para reconstruir índice vectorial
    st.markdown("### 🔄 Gestión de Datos")
    if st.button("Reconstruir Índice Vectorial", use_container_width=True):
        with st.spinner("Procesando PDFs de BimBam Buy en lotes y regenerando embeddings..."):
            try:
                import subprocess
                # Ejecutar vector_builder.py como proceso hijo
                result = subprocess.run(
                    [sys.executable, "vector_builder.py"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    st.success("¡Índice FAISS reconstruido con éxito!")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"Error al reconstruir índice: {result.stderr}")
            except Exception as e:
                st.error(f"Error inesperado: {e}")
                
    st.write("---")
    st.markdown(
        "<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>"
        "BimBam Buy Corp &copy; 2026<br>Oracle ONE / Alura Challenge"
        "</div>",
        unsafe_allow_html=True
    )

# ----------------- MAIN INTERFACE -----------------

# Header corporativo
st.markdown(
    '<div class="main-header">'
    '<h1>🤖 BimBam Buy Copilot IA</h1>'
    '<p>Asistente inteligente RAG de políticas internas, envíos, métodos de pago y garantías</p>'
    '</div>',
    unsafe_allow_html=True
)

# Fila superior de información de estado
col_status1, col_status2 = st.columns([1, 1])
with col_status1:
    st.markdown(
        'Estado del Agente: <span class="status-pill">🟢 En Línea / RAG Operativo</span>',
        unsafe_allow_html=True
    )
with col_status2:
    st.markdown(
        f"<div style='text-align: right; color: #64748b; font-size: 0.9rem;'>"
        f"Historial de chat: {len(st.session_state.messages)} mensajes"
        "</div>",
        unsafe_allow_html=True
    )

st.write("")

# Si el chat está vacío, mostrar pantalla de bienvenida y sugerencias
if not st.session_state.messages:
    st.markdown("### 👋 ¡Bienvenido, Colaborador!")
    st.markdown(
        "Consulta cualquier duda sobre el funcionamiento interno, políticas y operaciones. "
        "Haz clic en las siguientes tarjetas sugeridas para iniciar una conversación:"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.markdown(
            '<div class="suggestion-card">'
            '<h4>📦 Envíos y Tiempos</h4>'
            '<p>¿Cuánto cuesta el envío y cuáles son los plazos estimados para regiones?</p>'
            '</div>',
            unsafe_allow_html=True
        ):
            pass
            
    with col2:
        if st.markdown(
            '<div class="suggestion-card">'
            '<h4>💰 Reembolso y Devolución</h4>'
            '<p>¿Cuáles son los plazos y condiciones para un reembolso por insatisfacción?</p>'
            '</div>',
            unsafe_allow_html=True
        ):
            pass
            
    with col3:
        if st.markdown(
            '<div class="suggestion-card">'
            '<h4>🤝 Programa de Afiliados</h4>'
            '<p>¿Cuáles son las comisiones y cómo funciona la ventana de atribución de comisiones?</p>'
            '</div>',
            unsafe_allow_html=True
        ):
            pass
            
    st.write("")
    
    # Campo de selección rápida para Streamlit
    quick_q = st.selectbox(
        "O haz una pregunta rápida de la lista:",
        [
            "",
            "¿Cuáles son los plazos para solicitar un reembolso?",
            "¿Cómo funciona el programa de afiliados y qué comisiones se pagan?",
            "¿Qué métodos de pago son aceptados y cómo se validan?",
            "¿Cuál es el período de garantía y cobertura para productos electrónicos?"
        ],
        index=0
    )
    
    if quick_q:
        prompt = quick_q
        # Enviar inmediatamente
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Consultando base de conocimientos de BimBam Buy..."):
            response = rag_engine.query_rag(prompt, st.session_state.chat_history)
            answer = response.get("answer", "No se obtuvo respuesta.")
            context_docs = response.get("context", [])
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": context_docs
            })
            # Actualizar historial de LangChain
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=answer))
            st.rerun()

# ----------------- DISPLAY CHAT HISTORY -----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # Si tiene fuentes, mostrarlas de manera atractiva
        if msg.get("role") == "assistant" and msg.get("sources"):
            with st.expander("🔍 Ver Fuentes Consultadas (Documentos de Referencia)"):
                seen = set()
                for doc in msg["sources"]:
                    src = doc.metadata.get("source", "Desconocido")
                    page = doc.metadata.get("page", 0)
                    chunk_text = doc.page_content[:300] + "..."
                    
                    src_clean = os.path.basename(src)
                    source_key = f"{src_clean} (Pág. {page + 1})"
                    
                    if source_key not in seen:
                        st.markdown(
                            f'<div class="source-box">'
                            f'<div class="source-title">📄 {src_clean} - Página {page + 1}</div>'
                            f'<div class="source-content">"{chunk_text}"</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        seen.add(source_key)

# ----------------- INPUT AREA -----------------
if prompt := st.chat_input("Escribe tu pregunta sobre las políticas corporativas de BimBam Buy..."):
    # Renderizar inmediatamente la pregunta del usuario
    with st.chat_message("user"):
        st.write(prompt)
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generar respuesta
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Buscando en documentos de BimBam Buy..."):
            response = rag_engine.query_rag(prompt, st.session_state.chat_history)
            answer = response.get("answer", "No se obtuvo respuesta.")
            context_docs = response.get("context", [])
            
            response_placeholder.write(answer)
            
            # Mostrar fuentes
            if context_docs:
                with st.expander("🔍 Ver Fuentes Consultadas (Documentos de Referencia)"):
                    seen = set()
                    for doc in context_docs:
                        src = doc.metadata.get("source", "Desconocido")
                        page = doc.metadata.get("page", 0)
                        chunk_text = doc.page_content[:300] + "..."
                        
                        src_clean = os.path.basename(src)
                        source_key = f"{src_clean} (Pág. {page + 1})"
                        
                        if source_key not in seen:
                            st.markdown(
                                f'<div class="source-box">'
                                f'<div class="source-title">📄 {src_clean} - Página {page + 1}</div>'
                                f'<div class="source-content">"{chunk_text}"</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            seen.add(source_key)
            
    # Guardar en el historial de sesión
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": context_docs
    })
    
    # Mantener el historial de la conversación acotado (últimos 10 mensajes) para no desbordar el contexto del LLM
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=answer))
    
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]
        
    st.rerun()

# ----------------- ACCIONES DE HISTORIAL -----------------
if st.session_state.messages:
    st.write("")
    if st.button("Limpiar Conversación", type="secondary"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
