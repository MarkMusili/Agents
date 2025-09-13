# Customer Support Agent

An intelligent customer support agent built with LangGraph that automatically categorizes, analyzes sentiment, and routes customer queries to appropriate agents.

## Overview

This agent provides automated customer support by:
1. **Categorizing** customer queries into Technical, Billing, or General categories
2. **Analyzing sentiment** to detect negative emotions
3. **Routing** queries to specialized agents or escalating negative sentiment cases
4. **Generating** contextual responses based on the query type

## Architecture

The agent is built using LangGraph with a state-based workflow that processes customer queries through multiple stages:

```
Query → Categorize → Sentiment Analysis → Route → Agent → Response
                                            ↓
                                      Escalate (if negative)
```

### State Management

The agent maintains state using a Pydantic model with the following fields:
- `query`: The customer's original question/request
- `category`: Classification (Technical, Billing, General)
- `sentiment`: Sentiment analysis result (Positive, Neutral, Negative)
- `response`: Final generated response

## Features

### Query Categorization
Automatically classifies customer queries into:
- **Technical**: Software issues, bugs, troubleshooting, configuration problems
- **Billing**: Payment issues, subscriptions, invoices, pricing inquiries
- **General**: Product information, account management, policies, feedback

### Sentiment Analysis
Analyzes the emotional tone of customer messages to identify:
- Positive sentiment
- Neutral sentiment
- Negative sentiment (triggers escalation)

### Smart Routing
Routes queries based on category and sentiment:
- **Negative sentiment** → Escalates to human agent
- **Technical category** → Technical support agent
- **Billing category** → Billing support agent
- **General category** → General support agent

### Specialized Agents
Each agent provides tailored responses:
- **Technical Agent**: Troubleshooting steps, technical solutions
- **Billing Agent**: Payment assistance, account management
- **General Agent**: Product information, general assistance
- **Escalation Agent**: Professional escalation to human agents

## Dependencies

- `langgraph`: State graph workflow management
- `langchain-openai`: OpenAI integration
- `langchain-core`: Core LangChain functionality
- `pydantic`: Data validation and settings management
- `python-dotenv`: Environment variable management
