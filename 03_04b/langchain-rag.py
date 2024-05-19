# Import libraries
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain

# Define vectorstore
global vectorstore_faiss

# Define convenience functions


def config_llm():
    client = boto3.client('bedrock-runtime')

    model_kwargs = {
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9,
    }

    model_id = "meta.llama3-8b-instruct-v1:0"
    llm = Bedrock(model_id=model_id, client=client)
    llm.model_kwargs = model_kwargs
    return llm


def config_vector_db(filename):
    model_id = "amazon.titan-embed-image-v1"
    client = boto3.client('bedrock-runtime')
    bedrock_embeddings = BedrockEmbeddings(client=client, model_id=model_id)
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()
    vectorstore_faiss = FAISS.from_documents(pages, bedrock_embeddings)
    return vectorstore_faiss


def vector_search(query):
    docs = vectorstore_faiss.similarity_search_with_score(query)
    info = ""
    for doc in docs:
        info += doc[0].page_content+'\n'
    return info


# Configuring the llm and vector store
llm = config_llm()
vectorstore_faiss = config_vector_db("03_04b/social-media-training.pdf")

# Creating the template
my_template = """
Human: 
    You are a conversational assistant designed to help answer questions from an employee. 
    You should reply to the human's question using the information provided below. Include all relevant information but keep your answers short. Only answer the question. Do not say things like "according to the training or handbook or according to the information provided...".
    
    <Information>
    {info}
    </Information>
    

    {input}

Assistant:
"""

# Configure prompt template
prompt_template = PromptTemplate(input_variables=["info", "input"],
                                 template=my_template)

# Create llm chain
question_chain = LLMChain(
    llm=llm, prompt=prompt_template, output_key="generation")

# Get question, peform similarity search, invoke model and return result

while True:
    question = input("enter you question regarding the social media policy.\n")

    # get the context regarding the question
    info_ = vector_search(query=question)

    output = question_chain({"input": question, "info": info_})

    # print the output:
    print(f"\nAgent:\n{output['generation']}\n")
