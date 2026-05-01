from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAI
from getpass import getpass

api_key = getpass("Enter your Google API key: ")

llm = GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
print(
    llm.invoke(
        "What are some of the pros and cons of Python as a programming language?"
    )
)

##  python ./agentic/fundamental-langchain/02-setup-llm.py