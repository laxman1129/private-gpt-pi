import time
import os
from langchain_community.llms import LlamaCpp

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import (CSVLoader, PyMuPDFLoader)
from langchain_community.vectorstores import (DocArrayInMemorySearch, Chroma)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain_community.embeddings import HuggingFaceEmbeddings

import gradio as gr


def load_llm():
    return LlamaCpp(
            streaming = True,
            model_path="models/tiny-vicuna-1b.q5_k_m.gguf",
            n_gpu_layers=10,
            n_batch=512,
            temperature=0.1,
            top_p=1,
            verbose=True,
            n_ctx=4096
            )

def get_retriever(switch):
    results = []
    csv_loader = CSVLoader(file_path="source_docs/incidents.csv")
    pdf_loader = PyMuPDFLoader(file_path="source_docs/2375.pdf")
    results.extend(csv_loader.load())
    results.extend(pdf_loader.load())
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(results)
    if switch == 'memory':
        db = DocArrayInMemorySearch.from_documents(texts, embeddings)
    else :
        db = Chroma.from_documents(texts, embeddings, persist_directory="db")

    retriever = db.as_retriever()
    return retriever


def answer_question(question, chat_history):
    llm = load_llm()
    retriever = get_retriever('chroma')
    callback = [StreamingStdOutCallbackHandler()]
    qa_stuff = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        callbacks=callback,
        verbose = True
    )
    # query = "5 features of a350 aircraft"
    # start_time = time.time()
    # response = qa_stuff.invoke(input=query)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"\n\nElapsed time: {elapsed_time} seconds \n{response}")

    res = qa_stuff.invoke(question)
    answer = res['result']

    print(f"================>{res}")

    

    # for doc in sources:
    #     print(f"\n> " + doc.metadata['source'] + ":")
    #     print(doc.page_content)
    return answer


def start():    
    title = "Private GPT"
    examples = [
        "What is A380 Aircraft",
        "5 features of A380 aircraft"
    ]

    gr.ChatInterface(
        fn= answer_question,
        title=title,
        examples=examples
    ).launch()

start()