from langchain.agents import AgentType, Tool, create_csv_agent, initialize_agent
from langchain.chains import LLMChain, SimpleSequentialChain, TransformChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool, StructuredTool, Tool, YouTubeSearchTool, tool
from langchain.utilities import PythonREPL
from langchain.vectorstores import Chroma

import csv
import os

from datetime import datetime
from dotenv import load_dotenv

from brag_doc_loader import BragDocLoader, BragDocWriter
from fetch_data import fetch_second_worksheet

load_dotenv()

llm = OpenAI(temperature=0.5)

# Get user input

brag = input("What did you accomplish? ")

# examples = {"accomplishment": "Today, I helped Henry learn how to pass props"}

prompt_template = """You are a helpful assistant whose task is to help a software engineer
get track their progress in a brag document in order to get a promotion. Your task is to take
the accomplishment that the user gives you and make sure it sounds succint but detailed, humble, professional, and that
it's something an engineering manager would want to read. Do not add any additional facts, just stick to the ones the user provides you 
If the user's input is already well written, just return it as is. If the user does not supply a definite date, append this one: {date}

User's accomplishment: {brag}

Your spruced up version: """

prompt = PromptTemplate(input_variables=["brag", "date"], template=prompt_template)

# Pass input to initial prompt and return improved brag

rewrite_chain = LLMChain(llm=llm, prompt=prompt)



professional_brag = rewrite_chain.run(brag=brag, date=datetime.now().strftime("%d-%m-%Y"))

# Open brag doc file and parse questions and examples
    
# overall_chain = SimpleSequentialChain(chains=[rewrite_chain, categorize_chain], verbose=True)

# overall_chain.run(brag)

# Embed questions?

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
loader = BragDocLoader(filepath='Formatted Brag Doc - Jesse')

docs = loader.load()
db = Chroma.from_documents(docs, embeddings)

# index = VectorstoreIndexCreator().from_loaders([loader])

# query = f"Which of the brag document value prompts fits best to the following software engineering accomplishment? {professional_brag}"
# print(index.query(query))

# # print(docs)

# brag = "Today I finished writing a Notion document about how to improve our release process"

docs = db.similarity_search(professional_brag)

new_brag = Document(page_content=f"{professional_brag}", metadata={'row': docs[0].metadata['row']})
brag_doc_writer = BragDocWriter(filepath='Formatted Brag Doc - Jesse')
brag_doc_writer.write(new_brag)
