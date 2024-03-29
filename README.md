# Example Project for Retrieval Augmented Generation (RAG)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](https://github.com/tingsyo/example_rag/blob/main/README.md)
[![zh-TW](https://img.shields.io/badge/lang-zh-green.svg)](https://github.com/tingsyo/example_rag/blob/main/README.zh.md)

This is an example application to demenstrate [Retrieval Augmented Generation (RAG)](https://en.wikipedia.org/wiki/Prompt_engineering#Retrieval-augmented_generation) in [Traditional Chinese](https://en.wikipedia.org/wiki/Traditional_Chinese_characters).

We used [FastAPI](https://fastapi.tiangolo.com/) as the interface, [LangChain](https://www.langchain.com/) as the application framework, and [OpenAI-API](https://openai.com/blog/openai-api) as the LLM and embedding model provider. That is to say, you will need a valid OPENAI-API-KEY to run this example on its default settings. You can modify the code to use alternative models.

To avoid copyright issues, this demo provided RAG of the Bible. The [《國語和合本聖經》](https://www.translatebible.com/chinese_union_version.html) is a translation of the [English Revised Version](https://en.wikipedia.org/wiki/Revised_Version) of the Bible published in 1919 and is currently belong to the public domain.


## Getting Started

1. Clone this repo
```
git clone https://github.com/tingsyo/example_rag.git
```


2. Install dependecies
```
pip install -r requirements.txt
```


3. Run the app

Use the Bible for RAG by default:
```
python app.py
```

Or, you may specify your own vectorstore:
```
python app.py --vectorstore_path <PATH_TO_VECTORSTORE>
```


## Create vectorstore from self-owned data

We also provide an example script to convert all pdf files in the specified directory into a vectorstore. Usage:

```
python create_vectorstore_from_pdfs.py --data_path <PATH_TO_PDF_FILES> --output_path  <PATH_TO_OUTPUT_VECTORSTORE>
```

