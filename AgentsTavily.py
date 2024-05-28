import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
# Useful to add documents to the chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Useful to load the URL into documents
from langchain_community.document_loaders import WebBaseLoader
# Split the Web page into multiple chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Create Embeddings
from langchain_openai import OpenAIEmbeddings
# Vector Database FAISS
from langchain_community.vectorstores.faiss import FAISS
# USeful to create the Retrieval part
from langchain.chains import create_retrieval_chain
#Human Message and AI message
from langchain_core.messages import HumanMessage, AIMessage
# Retriever tool
from langchain.tools.retriever import create_retriever_tool

# Set the TAVILY API key directly
os.environ['TAVILY_API_KEY'] = 'tavily_key'

# Access the TAVILY API key from the environment variables
tavily_api_key= os.getenv('TAVILY_API_KEY')

# Create Retriever
loader = WebBaseLoader("https://python.langchain.com/docs/expression_language/")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
splitDocs = splitter.split_documents(docs)

embedding = OpenAIEmbeddings(api_key = openapikey)
vectorStore = FAISS.from_documents(docs, embedding=embedding)
retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

model = ChatOpenAI(
    model='gpt-3.5-turbo-1106',
    temperature=0.7,
    api_key = openapikey
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant called Max."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults()

retriever_tools = create_retriever_tool(
    retriever,
    "lcel_search",
    "Use this tool when searching for information about Langchain Expression Language (LCEL)."
)
tools = [search, retriever_tools]

agent = create_openai_functions_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools
)

def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return response["output"]

st.title("Document Search with RAG")
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
tavily_key = st.sidebar.text_input('Tavily API Key', type='password')
if not openai_api_key.startswith('sk-'):
   st.warning('Please enter your OpenAI API key!', icon='⚠')
if openai_api_key.startswith('sk-') and tavily_key:
   if __name__ == '__main__':
        chat_history = []
        while True:
          user_input = input("You: ")
          if user_input.lower() == 'exit':
              break

          response = process_chat(agentExecutor, user_input, chat_history)
          chat_history.append(HumanMessage(content=user_input))
          chat_history.append(AIMessage(content=response))
          st.write("Assistant:", response)
else:
   st.warning("Please enter your OpenAI API Key")
