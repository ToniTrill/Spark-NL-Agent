import os
import json
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

FAISS_PATH = "db/faiss_index" #a on es guarda la cache faiss
JSON_PATH='db/bird-1/dev.json' #on estan les pregutnes i respostes de la db


#inicialitzar embeding de google
def get_embeddings():
    return  GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                task_type="RETRIEVAL_QUERY",                
                dimensions=768,                             
            )
#crear la faiss excpete la pregunta actual cas UDF
def load_vector(db_id_array, udf_actual_question=None):
    embeddings = get_embeddings()
    #carregar o crear nova cache faiss
    if  os.path.exists(FAISS_PATH):
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else :
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, "r") as f:
                data= json.load(f)
            documents = []
            for item in data:
                if item["db_id"] in db_id_array and item["db_id"] != udf_actual_question:
                    doc = Document(
                        page_content=item["question"],
                        metadata={
                            "sql": item["SQL"],
                            "db_id": item["db_id"]
                        }
                    )
                    documents.append(doc)
            #crear ls db vectorial
            vector_store = FAISS.from_documents(documents, embeddings)
            #guardar en local per no recalcular cada evgada
            vector_store.save_local(FAISS_PATH)
            print("faiss guardat \n")
            return vector_store
        else:
            print(f"no hi ha el {JSON_PATH}")
            return None