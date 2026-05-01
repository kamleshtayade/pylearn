## langchain with Google API key with Env Variable
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

chain = prompt | llm

question = "How much is 2+2?"
print(chain.invoke({"question": question}))

"""
content="Let's think step by step.\nWe are asked to find the sum of 2 and 2.\n\n1.  Start with the first number: 2\n2.  Add the second number: + 2\n\nWhen you combine two units with another two units, you get a total of four units.\n\nSo, 2 + 2 = 4.\n\nThe answer is **4**." additional_kwargs={} response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 'safety_ratings': [], 'model_provider': 'google_genai'} id='lc_run--019ddd2a-ba2c-7f81-81e1-a71ab84cae20-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 22, 'output_tokens': 398, 'total_tokens': 420, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 313}}
"""

##  python ./agentic/fundamental-langchain/03-setup-prompt.py