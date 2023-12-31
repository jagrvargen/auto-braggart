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

load_dotenv()

# llm = ChatOpenAI(model="gpt-4-0613")
llm = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo-0613")

# brag = input("What did you accomplish? ")
brag = "I finished some jira tickets"

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

# OpenAI Functions
function_mapping = {
    "rewrite_chain": rewrite_chain.run,
    "categorize_chain": db.similarity_search,
    "jira_loader": jira_loader.load,
    "jira_chain": jira_chain.run,
    "brag_doc_loader": brag_doc_loader.load,
    "brag_doc_writer": brag_doc_writer.write,
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
            },
            {
                "name": "jira_loader",
                "description": "Use if and only if the user input includes the word 'jira'. This is to fetch their Jira tickets from the API.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "jira_chain",
                "description": "Use if and only if you have just used the Jira Loader. Use this to summarize the engineer's accomplishments.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticket": {
                            "type": "string",
                            "description": "the contents of the engineer's most recent Jira tickets",
                        }
                    },
                    "required": ["ticket"],
                },
            },
            {
                "name": "brag_doc_writer",
                "description": "Use when you've finished the engineer's accomplishments and need to write them to their brag document'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "new_brag": {
                            "type": "string",
                            "description": "The formatted accomplishment that needs to be recorded",
                        },
                    },
                    "required": ["new_brag"],
                },
            },
]

def call_func(func, **kwargs):
    return func(**kwargs)


# third_response = llm.predict_messages([HumanMessage(content=user_request),
#                                        AIMessage(content=str(first_response.additional_kwargs)),
#                                        AIMessage(content=str(second_response.additional_kwargs)),
#                                        ChatMessage(role='function',
#                                                     additional_kwargs = {'name': function_name},
#                                                     content = """
#                                                         Just made the following updates: 2022, opex -23 and
#                                                         Year: 2023
#                                                         Category: headcount
#                                                         Amount: 40
#                                                     """
#                                                    )
#                                        ],
#                                        functions=function_descriptions)
stop = 10
messages = [HumanMessage(content=brag)]
while True:

    message = llm.predict_messages(messages, functions=function_descriptions)
    # print(type(message), message, "\n")
    func = message.additional_kwargs['function_call']['name']
    kwargs = json.loads(message.additional_kwargs['function_call']['arguments'])
    if func == "rewrite_chain":
        kwargs['date'] = datetime.now().strftime("%d-%m-%Y")

    if func == "brag_doc_writer":
        call_func(function_mapping[message.additional_kwargs['function_call']['name']], **kwargs)
        break

    res = call_func(function_mapping[message.additional_kwargs['function_call']['name']], **kwargs)

    messages.append(AIMessage(content=str(message.additional_kwargs)))
    print(messages, "\n")
    stop -= 1
    if not stop:
        break