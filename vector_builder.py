import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Cargar variables de entorno
load_dotenv()

def build_vector_index():
    print("=== BimBam Buy - Constructor del Índice de Vectores ===")
    
    # Validar API Key de Google
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("ERROR: La variable de entorno GOOGLE_API_KEY no está configurada en el archivo .env.", file=sys.stderr)
        sys.exit(1)
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"ERROR: El directorio '{data_dir}' no existe. Asegúrate de tener los PDFs allí.", file=sys.stderr)
        sys.exit(1)
        
    # 1. Cargar PDFs
    print(f"Cargando archivos PDF desde el directorio '{data_dir}'...")
    loader = PyPDFDirectoryLoader(data_dir)
    try:
        documents = loader.load()
    except Exception as e:
        print(f"ERROR al cargar los PDFs: {e}", file=sys.stderr)
        sys.exit(1)
        
    if not documents:
        print("No se encontraron documentos en la carpeta 'data'. Por favor añade los PDFs correspondientes.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Se cargaron exitosamente {len(documents)} páginas de documentos.")

    # 2. Fragmentación de texto (Text Splitting)
    print("Dividiendo el texto en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total de fragmentos creados: {len(chunks)}")

    # 3. Generación de Embeddings y Vector Store (FAISS)
    print("Generando embeddings con Google (models/gemini-embedding-001) y construyendo el índice FAISS...")
    import time
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        # Procesamiento por lotes (Batching) para evitar Rate Limits (429) de la API gratuita
        batch_size = 15
        vector_store = None
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            print(f"Indexando lote de fragmentos {i+1} al {min(i+batch_size, len(chunks))}...")
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
            
            if i + batch_size < len(chunks):
                print("Esperando 15 segundos para respetar límites de cuota (RPM)...")
                time.sleep(15)
                
    except Exception as e:
        print(f"ERROR al generar embeddings o crear el índice FAISS: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Guardar Índice Localmente
    index_path = "faiss_index"
    print(f"Guardando el índice FAISS localmente en '{index_path}'...")
    try:
        vector_store.save_local(index_path)
        print("¡El índice vectorial de FAISS ha sido guardado exitosamente!")
        print("=== Proceso finalizado con éxito ===")
    except Exception as e:
        print(f"ERROR al guardar el índice localmente: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    build_vector_index()
