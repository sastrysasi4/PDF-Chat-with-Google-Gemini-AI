from dotenv import load_dotenv
load_dotenv()
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser

GOOGLE_API_KEY= os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

pdf_path = 'apple-privacy-policy.pdf'
loader = PyPDFLoader(file_path=pdf_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size = 1000,chunk_overlap = 30,separator='\n')
docs = text_splitter.split_documents(documents)

# creating and storing a vector DB
# vectorstore = FAISS.from_documents(docs, embeddings)
# vectorstore.save_local('faiss_db')

# loading vector DB
vectorstore = FAISS.load_local("faiss_db", embeddings)

retriever = vectorstore.as_retriever()

template = """Answer the question a a full sentence, based only on the following context:
{context}

Return you answer in three back ticks

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()
chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser


print(chain.invoke({"question": "summarize the given pdf"}))
