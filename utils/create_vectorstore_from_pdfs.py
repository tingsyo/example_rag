#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
This script creates vectorstore from PDF files located in the specified directory.
The default embedding model is OpenAIEmbeddings(), so a valid OPENAI_API_KEY is required.
'''
import os
import pypdf
import openai
import logging
import argparse
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

__author__ = "Ting-Shuo Yo"
__copyright__ = "Copyright 2024~, Akira Dialog Technology"
__credits__ = ["Ting-Shuo Yo"]
__license__ = "Apache License 2.0"
__version__ = "0.1.0"
__maintainer__ = "Ting-Shuo Yo"
__email__ = "tingyo@dataqualia.com"
__status__ = "development"
__date__ = '2024-02-20'

#-----------------------------------------------------------------
# Utility functions
#-----------------------------------------------------------------
def load_all_pdf_files(DATA_PATH):
    ''' Read in all pdf files with DataLoader '''
    # Load files one by one
    files = os.listdir(DATA_PATH)
    loaders = []
    for f in files:
        loaders.append(PyPDFLoader(os.path.join(DATA_PATH,f)))
    # Merge all pdf files as one loader
    loader_all = MergedDataLoader(loaders=loaders)
    # Split each pdf file into chuncks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", "。"],
        keep_separator=True,
    )
    pages = loader_all.load_and_split(text_splitter)
    return(pages)

#-----------------------------------------------------------------
# main function
#-----------------------------------------------------------------
def main():
    # Configure Argument Parser
    parser = argparse.ArgumentParser(description='To create vectorstore from the pdf files in the data_path')
    parser.add_argument('--data_path', '-i', type=str, required=True, help="Path of the PDF files.")
    parser.add_argument('--output_path', '-o', type=str, required=True, help='the path for the output vectorstore.')
    args = parser.parse_args()
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    # Use the provided arguments
    DATA_PATH = args.data_path
    OUTPUT_PATH = args.output_path
    # Read in pdf files
    docs = load_all_pdf_files(DATA_PATH)
    # Create vectorstore
    vstore = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())
    vstore.save_local(OUTPUT_PATH)
    return(0)


#==========
# Script
#==========
if __name__=="__main__":
    main()