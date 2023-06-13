from langchain.agents import AgentType, initialize_agent
from langchain.llms import OpenAI
from langchain.tools import Tool


from dotenv import load_dotenv



from fetch_data import fetch_second_worksheet

load_dotenv()

llm = OpenAI(temperature=0.0001)

tools = [
    Tool.from_function(
        func=fetch_second_worksheet,
        name = "FetchGoogleSheet",
        description="useful to fetch and save the google sheet you need before working on anything else"
    ),
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# agent.run(prompt)
