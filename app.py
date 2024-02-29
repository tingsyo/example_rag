# -*- coding: utf-8 -*-
import uvicorn
import openai
import logging
import argparse
from operator import itemgetter
from typing import List, Annotated

import faiss
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate


from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, Request, Response
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2024~, Akira Dialog Technology"
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2024-02-20'

'''
This app is a demo of Retrieval Augmented Generation (RAG).
'''
#-----------------------------------------------------------------
# Define global parameters
#-----------------------------------------------------------------
VECTOR_STORE_PATH = "./data/vectorstore/"
LLM_RAG = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
EMB_MOD = OpenAIEmbeddings()
#-----------------------------------------------------------------
# Define BSA
#-----------------------------------------------------------------
def invoke_RAG_chain(question):
    ''' Invoke the RAG chain. '''
    # Load vectorestore as the retriever
    vstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings=EMB_MOD)
    retriever = vstore.as_retriever()
    # Define prompts
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You're a helpful AI assistant. Given a user question and the context, \
             answer the user question in Traditional Chinese. If none of the \
             articles answer the question, just say you don't know.\n\n\
             Here is the context:{context}",
            ),
            ("human", "{question}"),
        ]
    )
    # Define answer
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])


    format = itemgetter("docs") | RunnableLambda(format_docs)
    # subchain for generating an answer once we've done retrieval
    answer = prompt | LLM_RAG | StrOutputParser()
    #Define chain
    chain = (
        RunnableParallel(question=RunnablePassthrough(), docs=retriever)
        .assign(context=format)
        .assign(answer=answer)
        .pick(["answer", "docs"])
    )
    # Invoke the RAG chain
    res = chain.invoke(question)
    logging.debug('Answer:\t'+res['answer'])
    logging.info('Sources:\t'+str([doc.metadata for doc in res['docs']]))
    # Done
    return(res)

def beautify_results(results, format='md'):
    ''' Format the results for display '''
    output=("## 您的提問：\n>> " + results['question']+"\n\n")
    output+=("## 查詢的結果：\n")
    output+=("### 摘要：\n>> "+results['answer'].replace('\n','\n>> ')+"\n\n")
    output+=("### 引用來源：\n")
    for source in results['citations']:
        output+=("- "+str(source).replace('\n','\n>> ')+"\n")
    return(output)

#-----------------------------------------------------------------
# Define FastAPI app
#-----------------------------------------------------------------
app = FastAPI()

# locate templates
templates = Jinja2Templates(directory="templates")
chat_log = []

@app.get("/", response_class=HTMLResponse)
async def bsa_page(request: Request):
    ''' GET: show the landing page as the interface '''
    return templates.TemplateResponse("home.html", {"request": request, "chat_log": chat_log})

@app.post("/", response_class=HTMLResponse)
async def bsa_run(request: Request, user_input: Annotated[str, Form()]):
    ''' POST: perform RAG and display the results. '''
    logging.info('[QUESTION]'+user_input)
    res_rag = invoke_RAG_chain(user_input)
    #logging.debug('[RESPONSE_RAG]'+res_rag)
    results = {
        'question': user_input,
        'answer': res_rag['answer'],
        'citations': [doc.metadata for doc in res_rag['docs']],
    }    
    response = beautify_results(results)
    logging.debug('[OUTPUT]'+response)
    chat_log.append(response)
    return templates.TemplateResponse("home.html", {"request": request, "chat_log": chat_log})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Example RAG Parameters')
    # Add arguments
    parser.add_argument('--vectorstore_path', type=str, required=False,
                        help="Path to vectorestore for retrival.", default=VECTOR_STORE_PATH)
    # Parse the arguments
    args = parser.parse_args()
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    # Use the provided arguments
    VECTOR_STORE_PATH = args.vectorstore_path
    #run server
    uvicorn.run(app)