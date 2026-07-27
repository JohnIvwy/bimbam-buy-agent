import sys
import rag_engine

def run_tests():
    questions = [
        "¿Cuáles son los plazos para solicitar un reembolso?",
        "¿Cómo funciona el programa de afiliados y qué comisiones se pagan?",
        "¿Cuánto cuesta el envío y cuáles son los tiempos de entrega?",
        "¿Cómo hago para hacer una lasaña de carne?"  # Pregunta fuera de contexto
    ]
    
    print("=== INICIANDO PRUEBAS DEL MOTOR RAG ===")
    
    for i, q in enumerate(questions, 1):
        print(f"\n--- Prueba {i}: {q} ---")
        response = rag_engine.query_rag(q)
        
        print("\n[Respuesta del Agente]:")
        print(response.get("answer"))
        
        print("\n[Fuentes consultadas]:")
        docs = response.get("context", [])
        if not docs:
            print("Ninguna fuente consultada.")
        else:
            seen_sources = set()
            for doc in docs:
                source = doc.metadata.get("source", "Desconocido")
                page = doc.metadata.get("page", 0)
                seen_sources.add(f"- {source} (Página {page + 1})")
            for src in seen_sources:
                print(src)
        print("-" * 50)

if __name__ == "__main__":
    run_tests()
