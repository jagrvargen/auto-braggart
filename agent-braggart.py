from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, ChatMessage
from langchain.tools import Tool, format_tool_to_openai_function
from langchain.vectorstores import Chroma

import json
import os

from datetime import datetime

from dotenv import load_dotenv

from brag_doc_loader import BragDocLoader, BragDocWriter
from jira_loader import JiraLoader
from tools.brag_doc_read_write_tool import CustomGoogleSheetsReaderTool, CustomGoogleSheetsWriterTool

load_dotenv()

llm = ChatOpenAI(model="gpt-4-0613")
# llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo-0613")
# llm = OpenAI(temperature=0.1)

# brag = input("What did you accomplish? ")
brag = "I fixed a bug"

prompt_template = """You are a helpful assistant whose task is to help a software engineer
get track their progress in a brag document in order to get a promotion. Your task is to take
the accomplishment that the user gives you and make sure it sounds succint but detailed, humble, professional, and that
it's something an engineering manager would want to read. Do not add any additional facts, just stick to the ones the user provides you 
If the user's input is already well written, just return it as is. If the user does not supply a definite date, append this one: {date}.
Once you've rewritten the accomplishment, you need to open their google sheets brag doc, assign the accomplishment to one of the
categories provided, then write to the doc.

User's accomplishment: {brag}

Your spruced up version: """

jira_template = """You are a helpful assistant whose task is to help a software engineer
get track their progress in a brag document in order to get a promotion. Your task is to 
read through their Jira tickets, and if they've completed one (the status of the ticket is 'Released'),
then summarize the work as an accomplishment that they can record. Include the ticket number.

The engineer's Jira ticket: {ticket}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["brag", "date"])

rewrite_chain = LLMChain(llm=llm, prompt=prompt)

jira_prompt = PromptTemplate(input_variables=["ticket"], template=jira_template)

jira_chain = LLMChain(llm=llm, prompt=jira_prompt)

brag_doc_loader = BragDocLoader()
docs = brag_doc_loader.load()
brag_doc_writer = BragDocWriter()

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
db = Chroma.from_documents(docs, embeddings)

jira_loader = JiraLoader()

tools = [
Tool(
    name = "rewrite_chain",
    func = rewrite_chain.run,
    description = "Use when you first receive a user's accomplishment and need to rewrite it to make it sound professional"
),
Tool(
    name = "categorize_chain",
    func = db.similarity_search,
    description = """Use after you've rewritten the user's accomplishment to categorize it according to the competency framework. 
    Output a JSON where the key 'page_content' has the value of the accomplishment, and the key 'metadata' maps to an object where the key 'row' maps to row
    of the first document returned from the categorization (found in the Document's metadata).
    """
),
Tool(
    name = "jira_loader",
    func = jira_loader.load,
    description = "Use if and only if the user input includes the word 'jira'. This is to fetch their Jira tickets from the API."
),
Tool(
    name = "jira_chain",
    func = jira_chain.run,
    description = "Use if and only if you have just used the Jira Loader. Use this to summarize their accomplishment",
    return_direct=True,
)
]

# OpenAI Functions
function_mapping = {
    "rewrite_chain": rewrite_chain.run,
    "categorize_chain": db.similarity_search,
}
function_descriptions = [
            {
                "name": "rewrite_chain",
                "description": "Use when you first receive a user's accomplishment and need to rewrite it to make it sound professional",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "brag": {
                            "type": "string",
                            "description": "The accomplishment that a user needs to have made more professional sounding",
                        },
                        "date": {
                            "type": "string",
                            "description": "The date that the accomplishment happened. Defaults to today's date if none is provided"
                        },
                    },
                    "required": ["brag"],
                },
            },
            {
                "name": "categorize_chain",
                "description": "Use after you've rewritten the user's accomplishment to categorize it according to the competency framework.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "professional_brag": {
                            "type": "string",
                            "description": "the accomplishment that needs to be categorized",
                        }
                    },
                    "required": ["professional_brag"],
                },
            }
]

def call_func(func, **kwargs):
    return func(**kwargs)

functions = [format_tool_to_openai_function(t) for t in tools]
# print(functions)
message = llm.predict_messages([HumanMessage(content=brag)], functions=function_descriptions)
# print(message)
print(json.loads(message.additional_kwargs['function_call']['arguments']))
kwargs = json.loads(message.additional_kwargs['function_call']['arguments'])
kwargs['date'] = datetime.now().strftime("%d-%m-%Y")
print(call_func(function_mapping[message.additional_kwargs['function_call']['name']], **kwargs))
exit()

# model = ChatOpenAI(temperature=0)
# planner = load_chat_planner(model)
# executor = load_agent_executor(model, tools, verbose=True)

# agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
# print(agent.run(brag))
# exit()

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
print(agent.agent.llm_chain.prompt.template)
output = agent.run(brag)

# page_content = 
print(f"NEW BRAG ~~> {output}")
# brag_doc_writer.write(new_brag)
