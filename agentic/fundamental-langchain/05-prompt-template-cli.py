## langchain with Google API key with Env Variable
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

import argparse

load_dotenv() # take environment variables from .env file

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
        Respond with format of:
        Topic Title : <Title>
        Content Outline:  <List of bullet points>
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
    "audience": parsed_args.audience
}

response = chain.invoke(variables)

print("Response from llm")
print("response content:")
print(response.content)
print("response metadata:")
print(response.metadata)

##  python ./agentic/fundamental-langchain/05-prompt-template-cli.py --topic "Agentic AI" --audience "aspiring AI professionals"