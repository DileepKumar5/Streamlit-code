# 🌟 Import Streamlit to build easy and fun web apps
import streamlit as st
# ⏳ Import time to use delays or timers if needed
import time
# 📄 Import a tool to open and read PDF files
from langchain_community.document_loaders import PyPDFLoader
# ✂️ Import a tool to split big text into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 🤖 Import Google AI tool to turn text into special number lists (embeddings)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# 🗂️ Import tool to store and search chunks of text quickly
from langchain_chroma import Chroma
# 💬 Import Google AI chat model to talk and answer questions
from langchain_google_genai import ChatGoogleGenerativeAI
# 🔗 Import tools to build chains that combine searching and answering
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# 📝 Import tool to create custom prompts for the AI chat
from langchain_core.prompts import ChatPromptTemplate



# 🌿 Import a tool to load secret keys and settings from a hidden file (.env)
from dotenv import load_dotenv

# 🔑 Load those secret keys and settings into the program so we can use them safely
load_dotenv()



# 🖥️ Show a big title on the web app page
st.title("RAG Application built on Gemini Model")


# 📄 Open and read the PDF file about Convolutional Neural Networks
loader = PyPDFLoader("An Introduction to Convolutional Neural Networks.pdf")

# 📚 Load all the text from the PDF into one big document
data = loader.load()

# ✂️ Cut the big document into smaller pieces, each about 1000 characters long,
# with a little overlap (100 characters) so pieces don’t miss important info
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 📑 Split the loaded PDF text into many smaller documents (chunks)
docs = text_splitter.split_documents(data)


# 🗂️ Create a smart storage that remembers and searches through all the text chunks using Google AI embeddings
vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

# 🔍 Make a search helper from that storage to find the 10 most similar text chunks when asked a question
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# 🤖 Create a Google AI chat model that will answer questions with low randomness (temperature=0 means very focused answers)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, max_tokens=None, timeout=None)



# 💬 Create a chat input box on the web page where the user can type a question
query = st.chat_input("Say something: ") 
prompt = query

# 🤖 Write instructions for the AI assistant on how to answer questions:
# - Use the info found in the text chunks (context)
# - If you don't know the answer, say so
# - Keep the answer short, max 3 sentences
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# 📝 Create a prompt template that tells the AI what the system says and what the human asks
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# 🔎 If the user typed a question in the chat box, then do the following:
if query:
    # 🤖 Create a chain that uses the AI to answer questions based on documents and our prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # 🔗 Create a Retrieval-Augmented Generation chain that uses the retriever and the Q&A chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 💬 Ask the AI model to answer the user’s question using the chain we created
    response = rag_chain.invoke({"input": query})

    # 🖥️ Show the user's question on the web page
    st.write("You asked:", query)
    
    # 🖥️ Show the AI's answer on the web page
    st.write("Answer:", response["answer"])


