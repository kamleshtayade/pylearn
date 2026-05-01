## langchain with Google API key with Env Variable
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv() # take environment variables from .env file

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

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
    "topic": "Agentic AI",
    "audience": "aspiring AI professionals"
}

response = chain.invoke(variables)

print("Response from llm")
print("response content: ", response.content)

##  python ./agentic/fundamental-langchain/04-prompt-template.py