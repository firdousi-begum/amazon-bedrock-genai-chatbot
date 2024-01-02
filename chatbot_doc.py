import time
import os
import streamlit as st
import uuid
from utils.langchain import LangChainAssistant
from utils.osretriever import OpenSearchAssistant
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
        header = "Amazon Bedrock Chatbot PDF"
        st.write(f"<h3 class='main-header'>{header}</h3>", unsafe_allow_html=True)
    with col3:
        clear = st.button("Clear Chat")

    return clear


clear = write_top_bar()

modelId="anthropic.claude-instant-v1"

keywords = [f'Model Id: {modelId}','Amazon Bedrock','Langchain', 'Vector Store: OpenSearch']
formatted_labels = [keyword_label(keyword) for keyword in keywords]
st.write(' '.join(formatted_labels), unsafe_allow_html=True)
apply_studio_style()


@st.cache_resource(ttl=1800)
def load_assistant():
    prompt_data = """You are a friendly, conversational car user manual assistant.
    You help user's understand the features of their cars, instructions on how to use it, answer questions and also help with troubleshooting steps in case of any issues or breakdown of car.
    You should ALWAYS answer user inquiries based on the context provided and avoid making up answers.
    If you don't know the answer, simply state that you don't know. Do NOT make answers and hyperlinks on your own.

    <context>
    {context}
    </context
    
    <question>{question}</question>"""
    
    os_assistant = OpenSearchAssistant(index_name = os.environ.get('OS_INDEX'))
    assistant = LangChainAssistant(modelId=modelId, retriever= os_assistant.retriever, prompt_data= prompt_data)

    return assistant, os_assistant

assistant, os_assistant = load_assistant()

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
        result  = assistant.chat_doc(prompt)

        message_placeholder.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})

# Sidebar with file uploader and button
with st.sidebar:
    st.header("Update Car Manual PDF Index")
    uploaded_file = st.file_uploader("Choose a PDF file (max 10 MB)", type=["pdf"], accept_multiple_files=False, key="file_uploader")
    
    if uploaded_file:
        if uploaded_file.size > 10 * 1024 * 1024:  # Check if file size exceeds 10MB
            st.error("File size exceeds the maximum limit of 10MB.")
        else:
            st.info("File uploaded successfully!\n\n Please NOTE 'Update Index' will replace existing data with new file.")
            if st.button("Update Index"):
                # Show a spinner while updating the index
                with st.spinner("Updating Index..."):
                    # Save the uploaded file temporarily
                    file_path = os.path.join("temp", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())

                    # Call the function to update the index with the file path
                    os_assistant.upload_doc_to_os(file_path)

                    # Remove the temporary file
                    os.remove(file_path)
                st.success("Index updated successfully!")