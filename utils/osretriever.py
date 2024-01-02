from requests.auth import HTTPBasicAuth
from langchain.schema import BaseMessage
import requests
import boto3
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import BedrockEmbeddings
from dotenv import load_dotenv
import os

class OpenSearchAssistant():
    def __init__(self, index_name):
        load_dotenv()
        self.boto3_bedrock  = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-west-2",
        )
        self.br_embeddings = BedrockEmbeddings(client=self.boto3_bedrock, model_id='amazon.titan-embed-text-v1')
        self.domain_index = index_name
        self.domain_endpoint = os.environ.get('OS_ENDPOINT')
        self.os_username = os.environ.get('OS_USERNAME')
        self.os_password = os.environ.get('OS_PASSWORD')
        self.retriever = self.get_vector_db().as_retriever(search_kwargs={"k": 5})
        

    def get_results(self):
        pass

    def _delete_index(self, domain_index, domain_endpoint, os_username, os_password):
        # Define a list of indices to delete
        indices_to_delete = [domain_index]  # Replace with your list of indices to delete

        # Iterate over the list of indices and send a DELETE request for each index
        for index in indices_to_delete:
            # Define the URL for the DELETE index API endpoint
            URL = f'{domain_endpoint}/{index}'
            # Send a DELETE request to the URL to delete the index
            response = requests.delete(URL, auth=HTTPBasicAuth(os_username, os_password))
            self.logger.info(response)
    
    def upload_doc_to_os(self, file_path: str):
        
        URL = f'{self.domain_endpoint}/{self.domain_index}'
        # Send a GET request to the URL to list all indices
        response = requests.get(URL, auth=HTTPBasicAuth(self.os_username, self.os_password))
        self.logger.info(response.text)

        # Check if the request was successful
        if response.status_code == 200:
            self._delete_index(self.domain_index, self.domain_endpoint, self.os_username, self.os_password)

        loader = PyPDFLoader(file_path)

        documents = loader.load() #

        text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=35)
        #text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400, separator=",")
        #chunks = text_splitter.split_documents(documents)


        print(f"documents:loaded:size={len(documents)}")
        pages=[]
        pages.extend(loader.load_and_split(text_splitter))
        print(f"Documents:after split and chunking size={len(pages)}")

        #self.vectordb.add_documents(pages)

        chunk_size = 50 # for OpenSearch Bulk API
        for i in range(0, len(pages), chunk_size):
            chunk = documents[i:i + chunk_size]
            print (f'Chunk to embed: {len(chunk)}')
            try:
                self.vectordb.add_documents(chunk)
            except:
                for doc in chunk:
                    try:
                        self.vectordb.add_documents(doc)
                    except:
                        print(f'Error chunk: {doc}')

    def get_vector_db(self):

        #print(self.domain_endpoint, self.domain_index, self.os_username, self.os_password)
        # vector store index
        vectordb = OpenSearchVectorSearch(
        opensearch_url=self.domain_endpoint,
        is_aoss=False,
        verify_certs = True,
        http_auth=(self.os_username, self.os_password),
        index_name = self.domain_index,
        embedding_function=self.br_embeddings)

        return vectordb
    
    # def get_retriever(self):
    #     retriever = OpenSearchRetriever(vectorstore= self.vectordb, search_type='similarity', search_kwargs={"k": 5})
    #     return retriever