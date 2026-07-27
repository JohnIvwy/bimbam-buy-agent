import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Cargar variables de entorno
load_dotenv()

# Validar API Keys
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not google_api_key:
    print("WARNING: GOOGLE_API_KEY no está configurada. Los embeddings de Google podrían fallar.")
if not groq_api_key:
    print("WARNING: GROQ_API_KEY no está configurada. El LLM de Groq fallará al realizar consultas.")

# Configuración del modelo y vector store
FAISS_INDEX_DIR = "faiss_index"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

_rag_chain = None

def get_rag_chain():
    """
    Inicializa y retorna la cadena RAG configurada con historial de conversación.
    Utiliza un patrón Singleton para evitar recargar el índice FAISS en cada consulta.
    """
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain

    # 1. Cargar embeddings
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    except Exception as e:
        print(f"Error al inicializar embeddings de Google: {e}", file=sys.stderr)
        raise e

    # 2. Cargar índice vectorial
    if not os.path.exists(FAISS_INDEX_DIR):
        raise FileNotFoundError(
            f"No se encontró el índice FAISS en '{FAISS_INDEX_DIR}'. "
            "Por favor ejecuta primero 'vector_builder.py' para construirlo."
        )
    
    try:
        # allow_dangerous_deserialization es seguro aquí porque el índice se construye localmente
        vector_store = FAISS.load_local(
            FAISS_INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"Error al cargar el índice FAISS: {e}", file=sys.stderr)
        raise e

    # 3. Configurar retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # 4. Configurar LLM de Groq
    groq_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    try:
        llm = ChatGroq(
            model=groq_model,
            temperature=0.0,
            groq_api_key=groq_api_key
        )
    except Exception as e:
        print(f"Error al inicializar ChatGroq ({groq_model}): {e}", file=sys.stderr)
        raise e

    # 5. Crear Prompt para re-formular la pregunta según el historial (History-aware retriever)
    contextualize_q_system_prompt = (
        "Dado el historial de chat y la última pregunta del usuario "
        "que podría hacer referencia al contexto del historial, "
        "formula una pregunta independiente que pueda ser entendida "
        "sin el historial de chat. NO respondas la pregunta, solo "
        "re-formúlala si es necesario, o devuélvela tal como está si ya es independiente."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 6. Crear Prompt del Sistema para responder preguntas basadas en contexto
    system_prompt = (
        "Eres el asistente virtual corporativo de BimBam Buy (E-commerce multiplataforma). "
        "Tu objetivo es responder a las consultas de los colaboradores sobre políticas de reembolso, "
        "envíos, métodos de pago, garantías y afiliados.\n\n"
        "REGLAS CRÍTICAS:\n"
        "1. Responde ÚNICAMENTE basándote en la información provista en el contexto a continuación.\n"
        "2. Si la información necesaria para responder la pregunta no está explícitamente en el contexto, "
        "debes responder exactamente con esta idea: 'Lo siento, como asistente corporativo de BimBam Buy "
        "no tengo información al respecto en mis documentos oficiales. Por favor contacta al área correspondiente.' "
        "Sé siempre educado y servicial.\n"
        "3. Mantén un tono formal, profesional, empático y cortés.\n"
        "4. Cita o menciona el documento y sección específica de donde proviene la información en tu respuesta si es posible.\n"
        "5. Bajo ninguna circunstancia inventes políticas, plazos, montos o condiciones que no estén en el contexto.\n\n"
        "Contexto disponible:\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 7. Crear cadena RAG completa
    _rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return _rag_chain

def query_rag(query: str, chat_history: list = None) -> dict:
    """
    Realiza una consulta a la cadena RAG de BimBam Buy.
    
    Args:
        query (str): La pregunta del usuario.
        chat_history (list): Historial de chat en formato LangChain (lista de BaseMessage o tuplas).
        
    Returns:
        dict: Diccionario que contiene 'answer' (respuesta del LLM) y 'context' (documentos recuperados).
    """
    if chat_history is None:
        chat_history = []
        
    chain = get_rag_chain()
    
    try:
        response = chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        return response
    except Exception as e:
        print(f"Error al ejecutar la consulta RAG: {e}", file=sys.stderr)
        return {
            "answer": f"Ocurrió un error interno al procesar tu consulta: {e}",
            "context": []
        }
