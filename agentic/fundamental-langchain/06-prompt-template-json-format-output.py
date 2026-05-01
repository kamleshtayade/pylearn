## langchain with Google API key with Env Variable
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

import argparse

load_dotenv() # take environment variables from .env file

# Define Pydantic model for output parsing
class ResponseModel(BaseModel):
    topic: str = Field(..., description="The topic of the learning outline")
    audience: str = Field(..., description="The audience for the learning outline")
    outline: list[str] = Field(..., description="The content outline")

# Define output parser using Pydantic model
# PydanticOutputParser expects the Pydantic model under the `pydantic_object` keyword
output_parser = PydanticOutputParser(pydantic_object=ResponseModel)

# format instruction for output parser
format_instructions = output_parser.get_format_instructions()



llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# argparse is a module in Python for parsing command-line arguments.
# In this script, we use argparse to define and parse the command-line arguments for the topic and audience.

args = argparse.ArgumentParser(description="Generate a learning outline on a given topic for a specific audience.")
args.add_argument("--topic", type=str, required=True, help="The topic to generate a learning outline for.")
args.add_argument("--audience", type=str, required=True, help="The audience to generate the learning outline for.")
parsed_args = args.parse_args()


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert educational content creator."),
        ("human", """ 
        Please create a detailed learning outline on the topic 
        of {topic} for {audience}.
        Respond in JSON format following these instructions:
        {format_instructions}
        Do not include any commentary or markdown code fences (for example, do NOT include ```json or ```).
        Adhere to JSON schema and field description strictly.
        """),
    ]
)

# first step - build prompt with variables

# second step - invoke the LLM with built prompt

chain = prompt | llm # create a chain by pipeing prompt to llm (LCEL syntax)

# LCEL syntax explain
# LCEL (LangChain Expression Language) allows you to create a pipeline of components by chaining them together with the `|` operator.
# In this example, we create a chain by piping the prompt to the LLM.

variables = {
    "topic": parsed_args.topic,
    "audience": parsed_args.audience,
    "format_instructions": format_instructions
}

response = chain.invoke(variables)

print("Response from llm")
print("response content:")
print(response.content)

# validate the response
#parsed_response = output_parser.parse(response.content)

#print("Parsed response:")
#print(parsed_response)

##  python ./agentic/fundamental-langchain/06-prompt-template-json-format-output.py --topic "Agentic AI" --audience "aspiring AI professionals"