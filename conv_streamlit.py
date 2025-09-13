import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


# 1. Define the state structure
class State(TypedDict):
    messages: Annotated[list, add_messages]


# 2. Create the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# 3. Define a simple node that calls the LLM
def chatbot_node(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# 4. Build the graph
graph = StateGraph(State)
graph.add_node("chatbot", chatbot_node)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

# 5. Compile the graph into an app
app = graph.compile()

# --- STREAMLIT APP STARTS HERE ---
st.title("🤖 LangGraph Chatbot")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Chat input box (like Streamlit native chat)
if user_input := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Invoke graph
    result = app.invoke({"messages": st.session_state.messages})
    bot_reply = result["messages"][-1].content

    # Add bot reply
    st.session_state.messages.append({"role": "assistant",
                                      "content": bot_reply})
    st.chat_message("assistant").write(bot_reply)
