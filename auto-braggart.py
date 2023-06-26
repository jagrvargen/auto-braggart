from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma

import os

from datetime import datetime
from dotenv import load_dotenv

from brag_doc_loader import BragDocLoader, BragDocWriter
from fetch_data import fetch_second_worksheet

load_dotenv()

gpt3_5 = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo-0613")
# llm = OpenAI(temperature=0.5)

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

brag_chain = LLMChain(llm=gpt3_5, prompt=brag_prompt)
professional_brag = brag_chain.run(brag=brag, date=datetime.now().strftime("%d-%m-%Y"))

# Rank the closest category to the brag and write to Google Sheets
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
loader = BragDocLoader(filepath='Blank Formatted Brag Doc - Jesse')
docs = loader.load()

# from langchain.embeddings import HuggingFaceInstructEmbeddings

# instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
#                                                       model_kwargs={"device": "cpu"})

db = Chroma.from_documents(docs, embeddings)
categories = db.similarity_search(professional_brag, k=3)

# new_brag = Document(page_content=f"{professional_brag}\n", metadata={'row': categories[0].metadata['row']})
# brag_doc_writer = BragDocWriter()
# brag_doc_writer.write(new_brag=new_brag, filepath='Blank Formatted Brag Doc - Jesse')



# Give the user more leeway to select the category
print(f'Select the most fitting category for your accomplishment -- "{professional_brag}"\n\n')
for i in range(len(categories)):
	print(f"{i+1}: {categories[i].page_content}\n")

choice = int(input("Pick 1, 2, or 3: "))
if choice not in (1,2,3):
	choice = int(input("Pick 1, 2, or 3: "))
new_brag = Document(page_content=f"{professional_brag}", metadata={'row': categories[choice-1].metadata['row']})
brag_doc_writer = BragDocWriter()
brag_doc_writer.write(new_brag, filepath='Blank Formatted Brag Doc - Jesse')



# Use LLM to do the reasoning instead?
# gpt_4 = ChatOpenAI(temperature=0.5, model="gpt-4-0613")

# final_template = """You are a helpful assistant whose task is to help a software engineer
# get track their progress in a brag document in order to get a promotion. Your task is to take
# the accomplishment that the user gives you, compare it to the three competency categories that
# are provided, and choose the best fit based on how well the accomplishment fits the desired
# competency.

# The accomplishment: {accomplishment}

# The competency categories: 
# {categories}

# Output the chosen category in backticks like so: `{{chosen category}}`
# """
# final_prompt = PromptTemplate(input_variables=["accomplishment", "categories"], template=final_template)
# final_chain = LLMChain(llm=gpt_4, prompt=final_prompt)
# print("\nFinal chain:\n")
# print(final_chain.run(accomplishment=professional_brag, categories=categories))
