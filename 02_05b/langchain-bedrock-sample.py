#Imports
import json
import boto3
from langchain.llms.bedrock import Bedrock

#Create the bedrock client
client = boto3.client("bedrock-runtime")
#setting model inference parameters
model_id = "amazon.titan-text-express-v1"
prompt = """as a hiring manager of a company, you need to write a welcoming mail to the new joinee. write the mail.
"""

model_kwargs = {
        "temperature": 0.1,  
        "topP": 1,
        "maxTokenCount": 256
    }

#Create the llm
my_llm = Bedrock(client=client, model_id=model_id, model_kwargs=model_kwargs)

#Generate the response
respose = my_llm.invoke(prompt)

#Display the result
print(f"generated text: \n{respose}")
