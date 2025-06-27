from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DATA_PATH="input_data/"

# Loading PDF

def load_pdf(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

docs=load_pdf(data=DATA_PATH)
print("No. of PDF pages: ", len(docs))


# Creating Text Chunks

def create_text_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

chunks=create_text_chunks(extracted_data=docs)
print("No. of Text Chunks: ", len(chunks))

# Creating Vector Embeddings 

def import_embedding_model():
    model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return model

model=import_embedding_model()

# Store embeddings in FAISS

DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(chunks, model)
db.save_local(DB_FAISS_PATH)