from pydantic import BaseModel
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import logging
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class State(BaseModel):
    query: str
    category: Optional[str] = None
    sentiment: Optional[str] = None
    response: Optional[str] = None


def load_prompt(filename: str) -> str:
    """Load prompt content from a file in the Prompts directory."""
    prompt_path = os.path.join(os.path.dirname(__file__), 'Prompts', filename)
    with open(prompt_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


def categorize(state: State) -> State:
    """Categorize the query into Technical, Billing, or General"""
    system_prompt = load_prompt("categorize.md")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Query: {query}")
    ])
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model
    category = chain.invoke({"query": state.query}).content

    logger.info(f"Categorized as: {category}")
    return {"category": category}

def sentiment_analysis(state: State) -> State:
    """Analyze the sentiment of the query"""
    system_prompt = load_prompt("sentiment_analysis.md")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Query: {query}")
    ])
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model
    sentiment = chain.invoke({"query": state.query}).content

    logger.info(f"Sentiment analyzed as: {sentiment}")
    return {"sentiment": sentiment}

def handle_technical_query(state: State) -> State:
    """Handle a technical query"""
    system_prompt = load_prompt("handle_technical.md")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Technical Issue: {query}")
    ])
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model
    response = chain.invoke({"query": state.query}).content

    logger.info("Technical query handled successfully")
    return {"response": response}

def handle_billing_query(state: State) -> State:
    """Handle a billing query"""
    system_prompt = load_prompt("handle_billing.md")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Billing Inquiry: {query}")
    ])
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model
    response = chain.invoke({"query": state.query}).content

    logger.info("Billing query handled successfully")
    return {"response": response}

def handle_general_query(state: State) -> State:
    """Handle a general query"""
    system_prompt = load_prompt("handle_general.md")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Customer Inquiry: {query}")
    ])
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model
    response = chain.invoke({"query": state.query}).content

    logger.info("General query handled successfully")
    return {"response": response}

def escalate(state: State) -> State:
    """Escalate the query to a human agent"""
    system_prompt = load_prompt("escalate.md")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Customer Concern (Requires Escalation): {query}")
    ])
    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model
    response = chain.invoke({"query": state.query}).content

    logger.info("Query escalated successfully")
    return {"response": response}

def route_query(state: State) -> str:
    """Route the query based on its sentiment and category."""
    if state.sentiment == "Negative":
        return "escalate"
    elif state.category == "Technical":
        return "handle_technical_query"
    elif state.category == "Billing":
        return "handle_billing_query"
    else:
        return "handle_general_query"

workflow = StateGraph(State)

workflow.add_node("categorize", categorize)
workflow.add_node("sentiment_analysis", sentiment_analysis)
workflow.add_node("handle_technical_query", handle_technical_query)
workflow.add_node("handle_billing_query", handle_billing_query)
workflow.add_node("handle_general_query", handle_general_query)
workflow.add_node("escalate", escalate)

workflow.add_edge("categorize", "sentiment_analysis")
workflow.add_conditional_edges(
    "sentiment_analysis",
    route_query,
    {
        "handle_technical_query": "handle_technical_query",
        "handle_billing_query": "handle_billing_query",
        "handle_general_query": "handle_general_query",
        "escalate": "escalate"
    }
)
workflow.add_edge("handle_technical_query", END)
workflow.add_edge("handle_billing_query", END)
workflow.add_edge("handle_general_query", END)
workflow.add_edge("escalate", END)

workflow.set_entry_point("categorize")

app = workflow.compile()

def chat(query: str) -> str:
    """Chat with the agent"""
    return app.invoke({"query": query})