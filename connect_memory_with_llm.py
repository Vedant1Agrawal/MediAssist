import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Setting up Mistral LLM from HuggingFace for generating responses

HF_TOKEN=os.environ.get("HF_TOKEN")
HF_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(hf_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=hf_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN, "max_length":"512"}
    )
    return llm

# Connecting LLM with FAISS and Creating chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Loading Database

DB_FAISS_PATH="vectorstore/db_faiss"
model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, model, allow_dangerous_deserialization=True)

# Creating Question-Answer Chain

qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HF_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Testing a single query

user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])