# Luna Jira Ticket Backend

## Description

This is an AI-powered Slack bot that uses LangGraph and OpenAI to automatically analyze team conversations and create Jira tickets from actionable work items mentioned in chat. The system bridges informal Slack discussions with formal Jira project tracking, intelligently assigning ticket properties like priority and type while avoiding duplicates.RetryClaude can make mistakes. Please double-check responses.

🛠️ Tech Stack:

Backend: Python FastAPI
Integration: Slack Bolt SDK
LLM: OpenAI


## Prerequisites

- Python 3.13

## Installation

1. Clone the repository:

```
git clone https://github.com/devinabia/lunatickets.git
```

2. Navigate to the project directory:


3. Create a virtual environment and activate it:

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

4. Install the required dependencies:

```
pip install -r requirements.txt
```

5. Set up the environment variables:

Create a `.env` file in the project root directory and add the variables given in env_sample:

## Running the Application

1. Start the FastAPI server:

```
uvicorn app.routes:app --reload
```

The server will start running at `http://localhost:8000`.

2. Access the application in your web browser:

```
http://localhost:8000
```

You should see the "server is up and running" message.

## Deployment

To deploy the application, you can use a hosting platform like Heroku, AWS, or DigitalOcean. Make sure to set the environment variables on the deployment platform as well.
