import os
import sys
import json
import asyncio
import random
import string
from uuid import uuid4
from typing import Any, Dict, List, Optional
from google.adk.agents.base_agent import BeforeAgentCallback
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import BeforeModelCallback
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
import yfinance as yf

# import pandas as pd
# import plotly.graph_objects as go
# import vertexai
# from google.colab import auth
from IPython.display import HTML, Markdown, display

# --- ADK, Agent, and Evaluation Components ---
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.events import Event
from google.adk.runners import Runner
import google.adk as adk
from google.adk.tools import agent_tool, google_search, tool_context
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types
from google.genai.types import Content, Part
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool


print("✅ All libraries are ready to go!")

# @title Set Your Google Cloud Project Details
PROJECT_ID = "adk-build-1"             # @param {type:"string"}
LOCATION = "us-central1"               # @param {type:"string"}

# Set environment variables for the ADK and gcloud
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

print(f"\n✅ Vertex AI configured for project '{PROJECT_ID}' in '{LOCATION}'.")

ticker_mapping = {
    "Apple": "AAPL",
    "Tesla": "TSLA"
}

INDIAN_STOCKS = {
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATAPOWER": "TATAPOWER.NS"
}

def get_stock_data(stock_symbol: str):
    """Get the stock data for a given stock symbol"""
    return yf.Ticker(stock_symbol).info

def stock_name_before_tool(callback_context: CallbackContext) :
    """Inspects/modifies the LLM request or skips the call."""
    agent_name = callback_context.agent_name
    print(f"[Callback] Before model call for agent: {agent_name}")
    print(f"[Callback] Callback state: {callback_context.state.to_dict()}")

# 1. Callback BEFORE the entire Agent execution starts
async def my_before_agent(callback_context: CallbackContext) -> Optional[Any]:
    print(f"--- [BEFORE AGENT] Executing agent: {callback_context.agent_name} ---")
    # You could perform authorization checks or pre-populate session state here
    return None  # Return None to allow the agent to proceed normally

# 2. Callback BEFORE the prompt is sent to the LLM (Model)
async def my_before_model(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    print(f"--- [BEFORE MODEL] Sending prompt to model: {llm_request} ---")
    # You can inspect/modify the prompt or skip the model call by returning an LlmResponse
    return None

# 3. Callback BEFORE a specific Tool is executed
async def my_before_tool(tool: Any, args: Dict[str, Any], tool_context: ToolContext) -> Optional[Dict[str, Any]]:
    print(f"--- [BEFORE TOOL] Tool '{tool.name}' called with args: {args} ---")
    # Useful for validating arguments or implementing a tool-level cache
    tool_name = tool.name
    if tool_name=='get_stock_data' and args.get('stock','').upper() in INDIAN_STOCKS:
        print(f"[Callback] Detected {args.get('stock', '').upper()}. Modifying arg to {INDIAN_STOCKS[args.get('country', '').upper()]}.")
        args['country'] = INDIAN_STOCKS[args.get('stock', '').upper()]
        print(f"[Callback] Modified args: {args}")
        return None
    return None

PROFANITY_LIST=["dangit", "fudge", "bing"]

def query_before_model_profanity_filter(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    """Inspects/modifies the LLM request or skips the call."""
    agent_name = callback_context.agent_name
    print(f"[Callback] Before model call for agent: {agent_name}")
    print(f"[Callback] Callback state: {callback_context.state.to_dict()}")

    # Inspect the last user message in the request contents
    last_user_message = ""
    if llm_request.contents and llm_request.contents[-1].role == 'user':
         if llm_request.contents[-1].parts:
            last_user_message = llm_request.contents[-1].parts[0].text
    print(f"[Callback] Inspecting last user message: '{last_user_message}'")

    for bad_word in PROFANITY_LIST:
        if bad_word.upper() in str(last_user_message).upper():
            print("[Callback] Profanity detected. Skipping LLM call.")
            # Return an LlmResponse to skip the actual LLM call
            # LlmResponse is interpretted as the actual LLM response
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="You kiss your mother with that mouth? LLM call was blocked by before_model_callback.")],
                )
            )

    print("[Callback] Query was clean. Proceeding with LLM call.")
    return None

stock_data_agent = Agent(
    name="stock_data_agent",
    model="gemini-2.5-flash",
    description="Fetch important data points and ratios of a given stock.",
    instruction="You are a stock research assistant. You are given a stock symbol and you need to return the company's pe ratio, market cap, pb ratio, ebidta.",
    tools=[get_stock_data],
    before_agent_callback=my_before_agent,
    before_model_callback=my_before_model,
    before_tool_callback=my_before_tool
)

web_search_agent = Agent(
    name="web_search_agent",
    model="gemini-2.5-flash",
    description="Find stock-related news and recent updates from the internet.",
    instruction="""
        role: Find stock-related news and recent updates from the internet.
        "Always include sources.",
        "Prefer official reports over general news.",
        "Summarize findings concisely."
        "Based on the sentiment in the summary return 'BUY' or 'SELL' as output."
    """,
    tools=[google_search],
    output_key="sentiment_analysis_result"
)

technical_analysis_agent = Agent(
    name="technical_analysis_agent",
    model="gemini-2.5-flash",
    description="Gives a buy or sell rating based on the technical anaylsis.",
    instruction="""
        role: Rate the stock buy or sell.
        "Randomly return the value 'BUY' or 'SELL'"
    """,
    output_key="technical_analysis_result"
)

fundamental_analysis_agent = Agent(
    name="fundamental_analysis_agent",
    model="gemini-2.5-flash",
    description="Gives a buy or sell rating based on the fundamental anaylsis.",
    instruction="""
        role: Rate the stock buy or sell.
        "Randomly return the value 'BUY' or 'SELL'"
    """,
    output_key="fundamental_analysis_result"
)


parallel_agent=ParallelAgent(
        name="parallel_agent",
        sub_agents=[web_search_agent,technical_analysis_agent,fundamental_analysis_agent]
    )

reporting_agent = Agent(
    name='reporting_agent', model='gemini-2.5-flash',
    instruction=f"""
        You are a helpful assistant. Combine the following results and provide a final verdict.
        'web_search_agent': {web_search_agent},
        'technical_analysis_agent': {technical_analysis_agent},
        'fundamental_analysis_agent': {fundamental_analysis_agent}
    """
)

orchestrator_agent = SequentialAgent(
    name='orchestrator_agent',
    sub_agents = [parallel_agent, reporting_agent],
    description="Workflow that finds multiple things that run in parallel and then sumarizes the results."
)



router_agent = Agent(
    name="router_agent",
    model="gemini-2.5-flash",
    description="Decides which agent or workflow to choose based on input query.",
    instruction="""
        You are a request router. Your job is to analyze a user's query and decide which of the following agents or workflows is best suited to handle it.
        Do not answer the query yourself, only return the name of the most appropriate choice. 

        Available Options:
        - 'web_search_agent': For queries *only* about the sentiment of the stock in the market and news.
        - 'stock_data_agent': For queries *only* about data points like the company's pe ratio, market cap, pb ratio, ebidta."
        - 'orchestrator_agent': For queries that ask about the sentiment, techincal and fundamental analysis"
        """,
    before_model_callback=query_before_model_profanity_filter
)

worker_agents = {
    'web_search_agent': web_search_agent,
    'stock_data_agent': stock_data_agent,
    'orchestrator_agent': orchestrator_agent
}

# Tools for Orchestrator agent

async def call_stock_data_agent(
    stock: str,
    tool_context: ToolContext
    ):
    agent_tool = AgentTool(stock_data_agent)
    stock_data_agent_output = await agent_tool.run_async(
        args={"request":stock}, tool_context=tool_context
    )
    # tool_context.state["retrieved_data"] = stock_data_agent_output
    return stock_data_agent_output

async def call_web_search_agent(
    stock: str,
    tool_context: ToolContext
    ):
    agent_tool = AgentTool(web_search_agent)
    agent_output = await agent_tool.run_async(
        args={"request": stock}, tool_context=tool_context
    )
    # tool_context.state["retrieved_data"] = agent_output
    return agent_output

async def call_agent_as_tool(
    agent: Agent,
    stock: str,
    tool_context: ToolContext
    ):
    agent_tool = AgentTool(agent)
    agent_output = await agent_tool.run_async(
        args={"request": stock}, tool_context=tool_context
    )
    # tool_context.state["retrieved_data"] = agent_output
    return agent_output


def create_stock_research_agent():
    """
    Research a stock and return a summary of the company's financials, industry trends, and market analysis.
    """
    # Create a new agent
    return Agent(
        name="stock_research_agent",
        model="gemini-2.5-flash",
        description="A agent that researches a stock and returns a summary of the company's financials, industry trends, and market analysis.",
        instruction="""You are a stock research head. You are given a stock symbol and you need to: 
        1. return the company's pe ratio, market cap, pb ratio, ebidta using 'call_stock_data_agent' tool and 
        2. Find stock-related news and recent updates from the internet
        3. Consolidate your analysis based on the information from step 2 and return it as a brief report along with values from step 1.
        """,
        tools=[call_stock_data_agent, call_web_search_agent],
        
    )

stock_research_agent = create_stock_research_agent()

print(f"Agent created: {stock_research_agent.name}")


async def run_agent(agent: Agent, query: str, session: Session, user_id: str, is_router: bool = False) -> str:
    """Initializes a runner and executes a query for an agent and sessiion"""

    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name=agent.name
    )

    final_response = ""

    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=Content(parts=[Part(text=query)], role="user"),
        ):
            if not is_router:
                print(f"EVENT: {event}")
                print(f"usage_metadata:{event.usage_metadata}")
            if event.is_final_response():
                final_response = event.content.parts[0].text

    except Exception as e:
        final_response = f"Error has occurred: {e}"

    if not is_router:
        print("\n"+"_"*50)
        print("Final response")
        print(final_response)
        print("\n"+"_"*50)

    return final_response

session_service = InMemorySessionService()
my_user_id = "stock_client_001"


async def run_stock_agent():
    research_session = await session_service.create_session(
        app_name=stock_research_agent.name,
        user_id=my_user_id
    )

    query = "AAPL"
    print("Stock being researched: ", query)

    # await run_agent(
    #     agent=stock_research_agent,
    #     query="AAPL",
    #     session=research_session,
    #     user_id=my_user_id
    # )

    router_session = await session_service.create_session(app_name=router_agent.name, user_id=my_user_id)
    chosen_route = run_agent(agent=router_agent, session=router_session, user_id=my_user_id, is_router=True)
    chosen_route = chosen_route.strip().replace("'", "").replace('"','').split(",")

async def run_fully_loaded_app():
    queries = [
        # Test Case 1: Simple Sequential Flow
        "Get me TATAPOWER pe ratio, market cap, pb ratio, ebidta."
        # "Get me the buy/sell rating of TSLA based on sentiment, techincal and fundamental analysis"
    ]

    for query in queries:
        print(f"\n{'='*60}\n🗣️ Processing New Query: '{query}'\n{'='*60}")

        # 1. Ask the Router Agent to choose the right agent or workflow
        router_session = await session_service.create_session(app_name=router_agent.name, user_id=my_user_id)
        print("🧠 Asking the router agent to make a decision...")
        chosen_route = await run_agent(router_agent, query, router_session, my_user_id, is_router=True)
        chosen_route = chosen_route.strip().replace("'", "")
        print(f"🚦 Router has selected route: '{chosen_route}'")

        # 2. Execute the chosen route
        if chosen_route in worker_agents:
            worker_agent = worker_agents[chosen_route]
            print(f"--- Handing off to {worker_agent.name} ---")
            worker_session = await session_service.create_session(app_name=worker_agent.name, user_id=my_user_id)
            await run_agent(worker_agent, query, worker_session, my_user_id)
            print(f"--- {worker_agent.name} Complete ---")
        else:
            print(f"🚨 Error: Router chose an unknown route: '{chosen_route}'")



if __name__ == "__main__":
    asyncio.run(run_fully_loaded_app())
