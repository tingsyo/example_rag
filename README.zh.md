# Retrieval Augmented Generation (RAG) 簡單範例
[![en](https://img.shields.io/badge/lang-en-blue.svg)](https://github.com/tingsyo/example_rag/blob/main/README.md)
[![zh-TW](https://img.shields.io/badge/lang-zh-green.svg)](https://github.com/tingsyo/example_rag/blob/main/README.zh.md)

本專案是一個簡單的繁體中文 [Retrieval Augmented Generation (RAG)](https://en.wikipedia.org/wiki/Prompt_engineering#Retrieval-augmented_generation) 系統。我們使用 [FastAPI](https://fastapi.tiangolo.com/) 提供界面，以 [LangChain](https://www.langchain.com/) 作為應用開發的基礎架構，而後端的語言模型則是使用 [OpenAI-API](https://openai.com/blog/openai-api)。這也意味著，如果要使用本應用的預設值，系統環境變數中需要一個有效的 OPENAI-API-KEY，當然您也可以修改程式碼使用其它的模型。 

為了避免版權爭議，本應用預設的文本是 [《國語和合本聖經》](https://www.translatebible.com/chinese_union_version.html)，這是 1919 出版的譯本，雖然這是百年前的翻譯，所用的詞語和語法跟現代有所不同，但它屬於公領域的文本。


## 使用方式

1. 下載本專案
```
git clone https://github.com/tingsyo/example_rag.git
```


2. 安裝相關套件
```
pip install -r requirements.txt
```


3. 啟動 RAG app

預設使用 《國語和合本聖經》 作為回答依據：
```
python app.py
```

也可以使用自定義的 vectorstore 來啟動 RAG：
```
python app.py --vectorstore_path <PATH_TO_VECTORSTORE>
```


## 將自有的文字內容轉換成 vectorstore

我們提供了一個將多個 pdf 檔案轉換成 vectorstore 的小工具，用法：

```
python create_vectorstore_from_pdfs.py --data_path <PATH_TO_PDF_FILES> --output_path  <PATH_TO_OUTPUT_VECTORSTORE>
```

