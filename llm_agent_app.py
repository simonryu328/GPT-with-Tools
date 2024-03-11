import os
from dotenv import load_dotenv
import streamlit as st
import langchain

from langchain_openai import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

load_dotenv()

st.set_page_config(page_title="LangChain Agents + MRKL")
st.title("LangChain Agents + MRKL")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

llm = OpenAI(temperature=0, streaming=True, openai_api_key = os.getenv("OPENAI_API_KEY"))
tools = load_tools(
    ["ddg-search"]
)
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if prompt := st.chat_input(placeholder="Who are the top 3 individual shareholders of Nvidia?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=openai_api_key,
        streaming=True
    )

    search_agent = initialize_agent(
        tools = [DuckDuckGoSearchRun(name="Search")],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st.write("Thinking...")
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_callback])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)