from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMChain
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool


from dotenv import load_dotenv

from brag_doc_loader import BragDocLoader, BragDocWriter
from tools.brag_doc_read_write_tool import CustomGoogleSheetsReaderTool, CustomGoogleSheetsWriterTool

load_dotenv()

llm = OpenAI(temperature=0.1)

# brag = input("What did you accomplish? ")
brag = "I fixed a bug"
filepath = "Formatted Brag Doc - Jesse"

prompt_template = """You are a helpful assistant whose task is to help a software engineer
get track their progress in a brag document in order to get a promotion. Your task is to take
the accomplishment that the user gives you and make sure it sounds succint but detailed, humble, professional, and that
it's something an engineering manager would want to read. Do not add any additional facts, just stick to the ones the user provides you 
If the user's input is already well written, just return it as is. If the user does not supply a definite date, append this one: {date}

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

brag_doc_loader = BragDocLoader()
brag_doc_writer = BragDocWriter()

tools = [Tool(
    name = "Rewrite Chain",
    func = rewrite_chain.run,
    description = "Use when you first receive a user's accomplishment"
),Tool(
    name = "Google Sheets Document Loader",
    func = brag_doc_loader.load,
    description = "Use when you need to fetch the user's Google Sheets document in order to read it"
), Tool(
    name = "Google Sheets Document Writer",
    func = brag_doc_writer.write,
    description = "Use when you need to fetch the user's Google Sheets document in order to write to it"
)]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run(f"Today I accomplished {brag} and I need to add it to my Google Sheets do. First I should open my Google sheets doc. I can find it in the filepath: `{filepath}` so that I can categorize the accomplishment before I try to write it.")
