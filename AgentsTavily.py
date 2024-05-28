import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Set up environment variables for API keys
st.title("Tavily Search with LangChain")

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
tavily_api_key = st.sidebar.text_input('Tavily API Key', type='password')

if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
elif openai_api_key.startswith('sk-') and tavily_api_key:
    os.environ['TAVILY_API_KEY'] = tavily_api_key

    # Load documents and split them
    loader = WebBaseLoader("https://python.langchain.com/docs/expression_language/")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)

    # Create embeddings and retriever
    embedding = OpenAIEmbeddings(api_key=openai_api_key)
    vectorStore = FAISS.from_documents(splitDocs, embedding=embedding)
    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    model = ChatOpenAI(
        model='gpt-3.5-turbo-1106',
        temperature=0.7,
        api_key=openai_api_key
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly assistant called Max."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # Set up Tavily search tool
    st.write("going to call search ")
    search = TavilySearchResults(api_key=tavily_api_key)
    st.write(search)

    retriever_tool = create_retriever_tool(
        retriever,
        "lcel_search",
        "Use this tool when searching for information about Langchain Expression Language (LCEL)."
    )
    
    tools = [search, retriever_tool]

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
        st.write(response["output"])
        return response["output"]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You: ", "")
    if st.button("Send"):
        if user_input:
            response = process_chat(agentExecutor, user_input, st.session_state.chat_history)
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=response))

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            st.write(f"You: {message.content}")
        elif isinstance(message, AIMessage):
            st.write(f"Assistant: {message.content}")
else:
    st.warning("Please enter your OpenAI API Key and Tavily API Key")
