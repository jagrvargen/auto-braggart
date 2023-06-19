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

chat_llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-0613")
llm = OpenAI(temperature=0.5)

# Get user input

brag = input("What did you accomplish? ")

# brag accomplishment chain
brag_template = """You are a helpful assistant whose task is to help a software engineer
get track their progress in a brag document in order to get a promotion. Your task is to take
the accomplishment that the user gives you and make sure it sounds succint but detailed, humble, professional, and that
it's something an engineering manager would want to read. Do not add any additional facts, just stick to the ones the user provides you 
If the user's input is already well written, just return it as is. If the user does not supply a definite date, append this one: {date}

User's accomplishment: {brag}

Your spruced up version: """

brag_prompt = PromptTemplate(input_variables=["brag", "date"], template=brag_template)

# Pass input to initial prompt and return improved brag

brag_chain = LLMChain(llm=llm, prompt=brag_prompt)

professional_brag = brag_chain.run(brag=brag, date=datetime.now().strftime("%d-%m-%Y"))

# Jira ticket chain
jira_template = """You are a helpful assistant whose task is to help a software engineer
get track their progress in a brag document in order to get a promotion. Your task is to 
read through their Jira tickets, and if they've completed one (the status of the ticket is 'Released'),
then summarize the work as an accomplishment that they can record. Include the ticket number.

The engineer's Jira ticket: {input}
"""
jira_prompt = PromptTemplate(input_variables=["input"], template=jira_template)

jira_chain = LLMChain(llm=llm, prompt=jira_prompt)

# Router chain
from langchain.chains.router import MultiPromptChain
from langchain.chains import ConversationChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

destination_chains = {'brag': brag_chain, 'jira': jira_chain}
prompt_infos = [
    {
        "name": "brag",
        "description": "Useful when the user has a generic accomplishment they would like to record",
        "prompt_template": brag_template,
    },
    {
        "name": "jira",
        "description": "Good only when the user mentions finishing a jira task. If they don't mention Jira, not useful.",
        "prompt_template": jira_template,
    },
]

default_chain = ConversationChain(llm=llm, output_key="text")

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)
print(chain.to_json())
# Open brag doc file and parse questions and examples
    
# overall_chain = SimpleSequentialChain(chains=[brag_chain, categorize_chain], verbose=True)

# overall_chain.run(brag)

# Embed questions?

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
loader = BragDocLoader()

docs = loader.load()
db = Chroma.from_documents(docs, embeddings)

# index = VectorstoreIndexCreator().from_loaders([loader])

# query = f"Which of the brag document value prompts fits best to the following software engineering accomplishment? {professional_brag}"
# print(index.query(query))

# # print(docs)

# brag = "Today I finished writing a Notion document about how to improve our release process"

docs = db.similarity_search(professional_brag)

new_brag = Document(page_content=f"{professional_brag}", metadata={'row': docs[0].metadata['row']})
brag_doc_writer = BragDocWriter()
brag_doc_writer.write(new_brag)
