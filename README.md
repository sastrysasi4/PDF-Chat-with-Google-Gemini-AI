
# PDF Chat with Google Gemini AI

In this project, we will see how to Chat with PDF using Gemini AI (Google Gen AI).
## Installation

These are the required packages

* Langchain
* Langchain-google-genai
* Pillow
* Python-dotenv
* Pypdf
* Faiss-cpu

## Documentation


Here,we are going to see how to store the user data. (PDF data) into vector stores and use RetrievalChain to ask questions about that data.


We have taken a PDF file that contains Apple privacy policy information. Then we passed this text data into a text splitter with a chunk size of 1000 and chunk overlap of 30. and we took embeddings and model from Google Gemini AI.


For the vector database, we chose FAISS, where we passed chunked text and OpenAI embeddings to change into a vector and store it. then using RetrievalChain to query the questions.
## Features

* Model -- Google Gemini AI
* Embeddings -- Google Gemini AI
* Document loader -- Pypdf loader
* Text splitter -- CharacterTextSplitter
* Vector DB -- Faiss




## Authors

- [@sasidhar](https://github.com/sastrysasi4)

