#Imports
import boto3
import json
#Create the bedrock client
client = boto3.client("bedrock-runtime")

#Setting the prompt
prompt = """Command: you are a freshman in college. write your assignment on genrative Ai.
Assignment:
"""
#Model specification
model_id = "amazon.titan-text-lite-v1"
#Configuring parameters to invoke the model
body = json.dumps({
  "inputText": prompt,
  "textGenerationConfig":{
    "maxTokenCount":256
  }
})
#Invoke the model
response = client.invoke_model(
  body=body, modelId=model_id
)

print(f"respose: \n{response}\n")
print(f"respose type: {type(response)}\n")

#Parsing and displaying the output

respose_body = json.loads(response.get("body").read())
generated_text = respose_body.get("results")[0].get("outputText")

print(f"generated text: \n{generated_text}")