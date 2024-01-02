# amazon-bedrock-genai-chatbot

This repository contains code samples for a chatbot using Amazon Bedrock, LangChain & OpenSearch.

### Gain Model Access from Amazon Bedrock Console

Visit the [Amazon Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) for instructions on gaining model access. For Claude access, use the `us-east-1` or `us-west-2` region. 

> NOTE: This codebase uses the region `us-west-2`. Please update the region in [langchain.py](utils/langchain.py) and [osretriever.py](utils/osretriever.py) if using another region.

### Install Required Packages

Open your terminal and run:

```bash
pip install -r requirements.txt
```

### Run Streamlit App for Simple Q&A Without Document

Run the following command:

```bash
streamlit run chatbot_simple.py
```

### Run Streamlit App for Chat Over Car Manual PDF Doc Using OpenSearch Index
1. Create an OpenSearch instance from the AWS console and note down the endpoint, username, password, and index.
2. Create a .env file in the root directory and add the following environment variables:

```env
OS_ENDPOINT="https://xxxxxx.region.es.amazonaws.com"
OS_USERNAME="REPLACE_WITH_YOUR_USERNAME"
OS_PASSWORD="REPLACE_WITH_YOUR_PASSWORD"
OS_INDEX="REPLACE_WITH_YOUR_INDEX"
```

3. Run the following command:

```bash
streamlit run chatbot_doc.py
```

4. Upload a Car Manual PDF with a size less than 10 MB to add the data to the OpenSearch index mentioned in the environment file.

### Start chatting
While chatting, see your terminal window to see how the chain is running.
> NOTE: Set verbose=False for chains `ConversationChain` & `ConversationalRetrievalChain` in the file [langchain.py](utils/langchain.py) if you dont want to see detail output.