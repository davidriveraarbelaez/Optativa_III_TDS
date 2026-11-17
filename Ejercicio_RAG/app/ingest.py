from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Leer documento
texto = Path(
    "/app/documentos/reglamento.txt"
).read_text(encoding="utf-8")

# 2. Segmentar documento
def dividir_texto(texto, tamano=300, overlap=50):
    chunks = []
    inicio = 0

    while inicio < len(texto):
        fin = inicio + tamano
        chunks.append(texto[inicio:fin])
        inicio += tamano - overlap

    return chunks

chunks = dividir_texto(texto)

print("Fragmentos generados:", len(chunks))

# 3. Cargar modelo de embeddings
modelo = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = modelo.encode(chunks).tolist()

# 4. Conectar con ChromaDB
cliente = chromadb.HttpClient(
    host="chroma",
    port=8000
)

coleccion = cliente.get_or_create_collection(
    name="reglamento"
)

# 5. Almacenar información
coleccion.upsert(
    ids=[f"doc_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print("Indexación completada")
print("Documentos almacenados:", coleccion.count())