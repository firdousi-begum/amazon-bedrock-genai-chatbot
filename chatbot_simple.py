import time

import streamlit as st
import uuid
from utils.langchain import LangChainAssistant
from utils.studio_style import apply_studio_style
from utils.studio_style import keyword_label


if "user_id" in st.session_state:
    user_id = st.session_state["user_id"]
else:
    user_id = str(uuid.uuid4())
    st.session_state["user_id"] = user_id

def write_top_bar():
    col1, col2, col3 = st.columns([2, 10, 3])
    with col1:
        st.image("images/amazon-bedrock-logo.svg", width=50)
    with col2:
        header = "Amazon Bedrock Chatbot Simple"
        st.write(f"<h3 class='main-header'>{header}</h3>", unsafe_allow_html=True)
    with col3:
        clear = st.button("Clear Chat")

    return clear


clear = write_top_bar()

modelId="anthropic.claude-v2"

keywords = [f'Model Id: {modelId}','Amazon Bedrock','Langchain']
formatted_labels = [keyword_label(keyword) for keyword in keywords]
st.write(' '.join(formatted_labels), unsafe_allow_html=True)
apply_studio_style()

@st.cache_resource(ttl=1800)
def load_assistant():
    assistant = LangChainAssistant(modelId=modelId)

    return assistant


def load_retriever():
    retriever = None

    return retriever


assistant = load_assistant()

if clear:
    st.session_state.messages = []
    assistant.clear_history()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # prompt = prompt_fixer(prompt)
        result, tokens_used  = assistant.chat(prompt)

        # # Simulate stream of response with milliseconds delay
        # for chunk in result.split():
        #     full_response += chunk + " "
        #     time.sleep(0.05)
        #     # Add a blinking cursor to simulate typing
        #     message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": full_response})