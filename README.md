# MediAssist - An AI Medical Chatbot

Tech stack: Python, LangChain, Hugging Face, FAISS, Mistral, Streamlit

Developed a medical chatbot using Retrieval-Augmented Generation (RAG) to provide accurate, context-aware health information through conversational AI.

Implemented FAISS (Facebook AI Similarity Search) to store and retrieve medical data efficiently, enabling scalable and fast similarity searches.

Utilized Hugging Face embeddings to encode medical documents into vector space for semantic retrieval.

Used LangChain for document ingestion, intelligent chunking, and text preprocessing to enhance context preservation.

Integrated the Mistral language model for generating coherent, human-like responses with strong contextual understanding.

## Steps to Set Up the Environment

### Install Required Packages
Run the following commands in your terminal:

```bash
install langchain langchain_community langchain_huggingface faiss-cpu pypdf
install huggingface_hub
install streamlit
```
### Launch Streamlit
Run the following command while in the MediAssist.py file

```bash
streamlit run MediAssist.py
```