
import chromadb

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from llm_provider import LLMProvider

load_dotenv()

# 1. Cargar modelo de embeddings
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Conectar con ChromaDB
db = chromadb.HttpClient(
    host="chroma",
    port=8000
)

collection = db.get_collection(
    name="reglamento"
)

# 3. Inicializar proveedor LLM
llm = LLMProvider()

# 4. Recibir pregunta
question = input("Escribe tu pregunta: ")

# 5. Generar embedding de la consulta
query_vector = embedding_model.encode(
    [question]
).tolist()

# 6. Recuperar documentos
results = collection.query(
    query_embeddings=query_vector,
    n_results=3
)

documents = results["documents"][0]

if not documents:
    print("No se encontró información.")
    raise SystemExit(0)

context = "\n\n".join(documents)

# 7. Construir prompt
prompt = f"""
Eres un asistente institucional.

Responde utilizando la información
proporcionada en el contexto.

Si no encuentras evidencia suficiente,
indica que no tienes información disponible.

No inventes requisitos ni procedimientos.

CONTEXTO:
{context}

PREGUNTA:
{question}
"""

# 8. Generar respuesta
answer = llm.generate(prompt)

print("\nRESPUESTA:")
print(answer)
