#Imports
import boto3
import json

#Create the client
client = boto3.client(service_name='bedrock-runtime')

# Define the user message to send.
user_message = """Translate the sentence from English to French.
    English: Learning about Generative Al is fun and exciting using Amazon Bedrock.
    French: """

# Embed the message in Llama 3's prompt format.
prompt = f"""
<|begin_of_text|>
<|start_header_id|>user<|end_header_id|>
{user_message}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

#Construct the body
#specify your prompt
body = json.dumps({
    "prompt": prompt, 
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9,
})

#Specify model id and content types
modelId = 'meta.llama3-8b-instruct-v1:0'
accept = 'application/json'
contentType = 'application/json'

#Invoke the model
response = client.invoke_model(
    body=body, 
    modelId=modelId, 
    accept=accept, 
    contentType=contentType
)

#Extract the response
response_body = json.loads(response.get('body').read())

#Display the output
print(response_body["generation"])