import streamlit as st
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from streamlit_chat_widget import chat_input_widget
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
client = OpenAI()


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

st.set_page_config(layout='wide')
st.markdown("<h1 style='text-align: center; color: white;'>🤖 AI Conversational Tutor</h1>",
            unsafe_allow_html=True)

st.markdown("""
<style>
/* Reduce Streamlit's default bottom padding */
.block-container {
    padding-bottom: 0rem !important;,
    padding-top: 0rem !important;
}

/* Adjust the iframe position */
iframe[title="streamlit_chat_widget.chat_input_widget"] {
    height: 80px !important;
    border-radius: 10px;
    margin-bottom: -15px;  /* pushes it closer to bottom */
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# --- Input Section ---
user_input = None
col1, col2 = st.columns([0.2, 0.8],
                        gap="Small",
                        vertical_alignment="top",
                        border=False,
                        width="stretch",
                        )

with col1:
    with st.container(horizontal_alignment='left',
                      vertical_alignment='top'):
        # st.header("A cat")
        st.image("https://th.bing.com/th/id/OIP.Xy3MEyqhqGeKjY5VznKpUgHaHa?w=160&h=180&c=7&r=0&o=7&dpr=1.5&pid=1.7&rm=3")
        st.button("Download", type="primary")

with col2:
    with st._bottom:
        with st.container(horizontal_alignment='left',
                          vertical_alignment='top',
                          border=None):
            user_input = chat_input_widget()

# Process the user's input from the widget
    if user_input:
        if "text" in user_input:
            user_input = user_input["text"]
        elif "audioFile" in user_input:
            audio_bytes = bytes(user_input["audioFile"])
            # st.audio(audio_bytes)
            audio_file = st.audio(audio_bytes)
            with open("recorded.wav", "wb") as f:
                f.write(audio_bytes)
                pass

            # Transcribe using Whisper
            with open("recorded.wav", "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            user_input = transcription.text
        st.session_state.messages.append({"role": "user",
                                          "content": user_input})
        st.chat_message("user").write(user_input)

        # Invoke graph
        result = app.invoke({"messages": st.session_state.messages})
        bot_reply = result["messages"][-1].content

        # Add bot reply
        st.session_state.messages.append({"role": "assistant",
                                          "content": bot_reply})
        st.chat_message("assistant").write(bot_reply)
