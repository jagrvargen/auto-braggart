from langchain.agents import AgentType, Tool, create_csv_agent, initialize_agent
from langchain.chains import LLMChain, SimpleSequentialChain, TransformChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool, StructuredTool, Tool, YouTubeSearchTool, tool
from langchain.utilities import PythonREPL
from langchain.vectorstores import Chroma

import os

from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(temperature=0.0001)

# Get user input

brag = input("What did you accomplish? ")

# examples = {"accomplishment": "Today, I helped Henry learn how to pass props"}

prompt_template = """You are a helpful assistant whose task is to help a software engineer
get track their progress in a brag document in order to get a promotion. Your task is to take
the accomplishment that the user gives you and make sure it sounds succint but detailed, humble, and that
it's something an engineering manager would want to read. Do not add any additional facts, just stick to the ones the user provides you 
If the user's input is already well written, just return it as is. Remove any references to time.

User's accomplishment: {brag}

Your spruced up version: """

prompt = PromptTemplate(input_variables=["brag"], template=prompt_template)

# Pass input to initial prompt and return improved brag

rewrite_chain = LLMChain(llm=llm, prompt=prompt)

# cleaned_brag = llm_chain.run(brag)

# Open brag doc file and parse questions and examples

brag_doc = {
    "Impact Over Effort": {
        "You are autonomous regarding industry basics and how code interactions work.": "",
        "You are autonomous (*1) for low complexity problems, defined by other engineers in the squad and impact domains owned by your squad. " +
        "(*1) Autonomy at this level means considering alternative approaches when blocked, anticipating risks and challenges for your commitments, balancing pragmatism and quality and driving the implementation of the feature. ": "",
        "You are starting to identify opportunities to improve features, or to create new ones that can generate a positive impact for our customers.": "",
	},
    "Discpline Excellence": {
        "You have some expertise in some domains that allows you to perform autonomously in them.": "",
        "You require support with unfamiliar domains and tasks. You factor your domain knowledge into the execution of tasks.": "",
        "You produce new instances, or minor improvements, to the existing architecture for your discipline, demonstrating nuanced understanding on your discipline constraints within your team domains.": "",
        "You autonomously adapt to problematic or challenging situations affecting your daily work in an appropriate manner, understanding that business requirements change regularly and that are no perfect scenarios.": "",
        "You understand your current team domains and metrics, and you start to lead, with support from the rest of the team, on firefighting issues.": ""
	},
    "People And Career": {
        "You are present and provide your inputs in activities supporting team operational health, by respecting and participating in your team dynamics and ceremonies.": "",
        "You provide feedback on how to improve existing processes and proposes changes that positively affect your local team.": "",
        "You provide feedback to team members, especially newbies, to help them to get onboarded in your team.": "",
	},
    "Be a Good Person - Communicate Effectively": {
        "You share details around your work with more senior engineers in the team, by raising your hand of difficulties or asking for direction / guidance, when needed.": "",
        "You focus your conversations with your team leads (PM / EM / senior engineers) or other stakeholders in your team (design, data, etc) on timelines, commitments and blockers.": "",
        "You collaborate submitting PRs that follow the existing guidelines and engineering KPIs providing enough context to help reviewers get up to speed. You collaborate reviewing and giving meaningful comments in others PR’s inside your team.": "",
	},
}

values_prompt_template = """You are a helpful assistant whose task is to help a software engineer
get track their progress in a brag document in order to get a promotion. Your task is to take
the accomplishment that the user gives you and categorize it according to one of the following prompt questions and its associated value.
    Value: Impact Over Effort
    
    Prompt Questions:
     	"You are autonomous regarding industry basics and how code interactions work."
        "You are autonomous (*1) for low complexity problems, defined by other engineers in the squad and impact domains owned by your squad."
        "(*1) Autonomy at this level means considering alternative approaches when blocked, anticipating risks and challenges for your commitments, balancing pragmatism and quality and driving the implementation of the feature.
        "You are starting to identify opportunities to improve features, or to create new ones that can generate a positive impact for our customers."

    Value: Discipline Excellence
    
    Prompt Questions:
     	"You have some expertise in some domains that allows you to perform autonomously in them."
        "You require support with unfamiliar domains and tasks. You factor your domain knowledge into the execution of tasks."
        "You produce new instances, or minor improvements, to the existing architecture for your discipline, demonstrating nuanced understanding on your discipline constraints within your team domains."
        "You autonomously adapt to problematic or challenging situations affecting your daily work in an appropriate manner, understanding that business requirements change regularly and that are no perfect scenarios."
        "You understand your current team domains and metrics, and you start to lead, with support from the rest of the team, on firefighting issues."

    Value: People And Career
    
    Prompt Questions:
     	"You are present and provide your inputs in activities supporting team operational health, by respecting and participating in your team dynamics and ceremonies."
        "You provide feedback on how to improve existing processes and proposes changes that positively affect your local team."
        "You provide feedback to team members, especially newbies, to help them to get onboarded in your team."

    Value: Be a Good Person - Communicate Effectively
    
    Prompt Questions:
     	"You share details around your work with more senior engineers in the team, by raising your hand of difficulties or asking for direction / guidance, when needed."
        "You focus your conversations with your team leads (PM / EM / senior engineers) or other stakeholders in your team (design, data, etc) on timelines, commitments and blockers."
        "You collaborate submitting PRs that follow the existing guidelines and engineering KPIs providing enough context to help reviewers get up to speed. You collaborate reviewing and giving meaningful comments in others PR’s inside your team."
        
    The user's input {brag}
    
    The prompt it fits best with: """

values_prompt = PromptTemplate(input_variables=["brag"], template=values_prompt_template)
categorize_chain = LLMChain(llm=llm, prompt=values_prompt)

overall_chain = SimpleSequentialChain(chains=[rewrite_chain, categorize_chain], verbose=True)

# overall_chain.run(brag)

# Embed questions?

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
# # loader = CSVLoader(file_path='./data/formatted-brag-doc.csv')
loader = UnstructuredExcelLoader(file_path='./data/formatted-brag-doc.xlsx')

# docs = loader.load()

from langchain.indexes import VectorstoreIndexCreator

index = VectorstoreIndexCreator().from_loaders([loader])

query = f"Which of the brag document value prompts fits best to the following software engineering accomplishment? (Prompts start with the word 'You') {brag}"
print(index.query(query))

# # print(docs)

# db = Chroma.from_documents(docs, embeddings)

# brag = "Today I finished writing a Notion document about how to improve our release process"

# docs = db.similarity_search(brag)

# print(docs[0].page_content)



# Create example selector

# Create few shot template

# Compare brag with semantic similarity to Qs

# Fill in document and save
