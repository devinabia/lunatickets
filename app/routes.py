from fastapi import FastAPI, APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from .views import JiraService
from .schemas import UserQuery
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
import logging
import requests
import asyncio
import re
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize Slack app
slack_app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
)


def call_jira_api(query, channel_id=None):
    print(channel_id, ".......")
    """Helper function to call Jira API with optional channel ID"""
    try:
        api_endpoint = f"{os.getenv('APP_BACKEND_URL')}ask-query"

        # Prepare payload with channel_id if provided
        payload = {"query": query}
        if channel_id:
            payload["channel_id"] = channel_id

        response = requests.post(
            api_endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("data", "No response from Jira")
            else:
                return f"Error: {result.get('message', 'Unknown error occurred')}"
        else:
            return f"API Error: {response.status_code}"
    except Exception as e:
        logger.error(f"Error calling Jira API: {e}")
        return "Something went wrong. Please try again."


# Slack Event Handlers
@slack_app.event("app_mention")
def handle_app_mention(event, say):
    """Handle when bot is mentioned in channels for Jira operations"""
    try:
        logger.info(f"App mention received: {event}")
        text = event.get("text", "")
        channel_id = event.get("channel")  # Extract channel ID
        user_query = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

        # Log channel information
        logger.info(f"Message from channel: {channel_id}")

        if not user_query:
            say(
                "Please provide a Jira request after mentioning me. Example: `@jirabot create a bug in AI project for login issues`"
            )
            return

        # Pass channel_id to the API call
        answer = call_jira_api(user_query, channel_id)
        say(
            {
                "text": f"{answer}",
                "unfurl_links": True,
                "unfurl_media": True,
            }
        )

    except Exception as e:
        logger.error(f"Error handling app mention: {e}")
        say("Something went wrong. Please try again.")


@slack_app.message("")
def handle_dm_message(message, say):
    """Handle direct messages to the bot"""
    try:
        if message.get("channel_type") != "im":
            return

        logger.info(f"DM received: {message}")
        user_query = message["text"].strip()
        channel_id = message.get("channel")  # Extract channel ID (DM channel)

        # Log channel information
        logger.info(f"Direct message from channel: {channel_id}")

        if not user_query:
            say(
                "Please provide a Jira request. Example: `create a task in AI project for user authentication`"
            )
            return

        if user_query.lower().startswith("ask"):
            return

        # Pass channel_id to the API call
        answer = call_jira_api(user_query, channel_id)
        say(f"{answer}")

    except Exception as e:
        logger.error(f"Error handling DM message: {e}")
        say("Something went wrong. Please try again.")


@slack_app.message("create")
def handle_create_message(message, say):
    """Handle messages starting with 'create'"""
    try:
        logger.info(f"Create message received: {message}")
        user_query = message["text"]
        channel_id = message.get("channel")  # Extract channel ID

        # Log channel information
        logger.info(f"Create message from channel: {channel_id}")

        # Pass channel_id to the API call
        answer = call_jira_api(user_query, channel_id)
        say(
            {
                "text": f"{answer}",
                "unfurl_links": True,
                "unfurl_media": True,
            }
        )

    except Exception as e:
        logger.error(f"Error handling create message: {e}")
        say("Something went wrong. Please try again.")


@slack_app.command("/jira")
def handle_jira_slash_command(ack, respond, command):
    """Handle /jira slash command"""
    ack()

    try:
        logger.info(f"Jira slash command received: {command}")
        query = command["text"].strip()
        channel_id = command.get("channel_id")  # Extract channel ID

        # Log channel information
        logger.info(f"Slash command from channel: {channel_id}")

        if not query:
            respond(
                {
                    "response_type": "ephemeral",
                    "text": "Please provide a Jira request. Example: `/jira create a bug in AI project for database issues`",
                }
            )
            return

        # Pass channel_id to the API call
        answer = call_jira_api(query, channel_id)
        respond(
            {
                "response_type": "in_channel",
                "text": f"{answer}",
                "unfurl_links": True,
                "unfurl_media": True,
            }
        )

    except Exception as e:
        logger.error(f"Error handling jira slash command: {e}")
        respond(
            {
                "response_type": "ephemeral",
                "text": "Something went wrong. Please try again.",
            }
        )


# Create Slack handler
slack_handler = SlackRequestHandler(slack_app)


class BotRouter:
    def __init__(self):
        self.jira_service = JiraService()
        self.router = APIRouter()
        self.setup_routes()

    def setup_routes(self):
        self.router.add_api_route(
            "/ask-query",
            self.handle_ask_query,
            methods=["POST"],
        )

        self.router.add_api_route(
            "/test-jira-query",
            self.jira_service.process_query,
            methods=["POST"],
        )

    async def handle_jira_query(self, request: Request):
        """Handle Jira-specific queries"""
        try:
            content_type = request.headers.get("content-type", "").lower()

            if "application/json" in content_type:
                body = await request.json()
                user_query = body.get("query")

                if not user_query:
                    raise HTTPException(status_code=422, detail="Missing 'query' field")

                result = self.jira_service.process_query(user_query)
                return result

            else:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported content type. Use application/json",
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in handle_jira_query: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def handle_ask_query(
        self, request: Request, background_tasks: BackgroundTasks
    ):
        """
        Unified endpoint that handles both JSON (API calls) and form data (Slack slash commands)
        """
        try:
            content_type = request.headers.get("content-type", "").lower()
            logger.info(f"Received request with content-type: {content_type}")

            if "application/json" in content_type:
                logger.info("Processing JSON request")
                try:
                    body = await request.json()
                    if "query" not in body:
                        raise HTTPException(
                            status_code=422, detail="Missing 'query' field"
                        )

                    # Extract channel_id if provided in JSON payload
                    channel_id = body.get("channel_id")
                    if channel_id:
                        logger.info(f"API request from channel: {channel_id}")

                    user_query = UserQuery(query=body["query"])

                    # Pass channel_id to the service
                    result = await self.jira_service.process_query(
                        user_query, channel_id
                    )

                    return result

                except Exception as e:
                    logger.error(f"JSON processing error: {e}")
                    raise HTTPException(status_code=400, detail=str(e))

            elif "application/x-www-form-urlencoded" in content_type:
                logger.info("Processing Slack form data request")

                try:
                    form_data = await request.form()
                    command = form_data.get("command", "")
                    text = form_data.get("text", "").strip()
                    user_name = form_data.get("user_name", "Unknown")
                    response_url = form_data.get("response_url", "")
                    channel_id = form_data.get("channel_id", "")  # Extract channel ID

                    logger.info(
                        f"Slack - Command: {command}, Text: {text}, User: {user_name}, Channel: {channel_id}"
                    )
                    logger.info(f"Response URL: {response_url}")

                    if command != "/jira":
                        return {
                            "response_type": "ephemeral",
                            "text": f"Unknown command: {command}",
                        }

                    if not text:
                        return {
                            "response_type": "ephemeral",
                            "text": "Please provide a question. Example: `/jira what is the deployment process?`",
                        }

                    if not response_url:
                        logger.error("No response_url provided by Slack")
                        return {
                            "response_type": "ephemeral",
                            "text": "Invalid request from Slack",
                        }

                    background_tasks.add_task(
                        self.process_slack_query_async,
                        text,
                        user_name,
                        response_url,
                        channel_id,
                    )

                    logger.info("Returning immediate acknowledgment to Slack")
                    return {
                        "response_type": "in_channel",
                        "text": f"{user_name} asked: {text}",
                    }

                except Exception as e:
                    logger.error(f"Slack processing error: {e}")
                    return {
                        "response_type": "ephemeral",
                        "text": "Something went wrong. Please try again.",
                    }

            else:
                raise HTTPException(
                    status_code=415, detail=f"Unsupported content type: {content_type}"
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"General error in handle_ask_query: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    async def process_slack_query_async(
        self, query: str, user_name: str, response_url: str, channel_id: str = None
    ):
        """
        Background task to process the actual query and send response to Slack
        """
        try:
            logger.info(
                f"Starting background processing for query: {query} from channel: {channel_id}"
            )

            await asyncio.sleep(1)

            user_query = UserQuery(query=query)
            # Pass channel_id to the service
            result = await self.jira_service.process_query(user_query, channel_id)
            logger.info(f"Query processed successfully, sending response to Slack")

            final_response = {
                "response_type": "in_channel",
                "replace_original": True,
                "text": f"Jira Ticket\n\n{result.get('data', '')}",
            }

            response = requests.post(
                response_url,
                json=final_response,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 200:
                logger.info(
                    f"Successfully sent final response to Slack for query: {query}"
                )
            else:
                logger.error(
                    f"Failed to send response to Slack. Status: {response.status_code}, Response: {response.text}"
                )

        except Exception as e:
            logger.error(f"Error in background task: {e}", exc_info=True)

            error_response = {
                "response_type": "in_channel",
                "replace_original": True,
                "text": "Sorry, something went wrong while processing your question. Please try again.",
            }

            try:
                error_result = requests.post(
                    response_url,
                    json=error_response,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                if error_result.status_code == 200:
                    logger.info("Successfully sent error response to Slack")
                else:
                    logger.error(
                        f"Failed to send error response to Slack: {error_result.status_code}"
                    )
            except Exception as send_error:
                logger.error(f"Failed to send error response to Slack: {send_error}")


def create_app():
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="Jira Bot API",
        description="API for Jira operations with Slack integration",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Create router instance
    bot_router = BotRouter()

    # Include router with prefix (keeping your original structure without prefix)
    app.include_router(bot_router.router, tags=["Bot"])

    # Add Slack routes
    @app.post("/slack/events", tags=["Slack"])
    async def slack_events(request: Request):
        """Handle Slack events"""
        try:
            logger.info(f"Slack event received: {request.method} {request.url}")
            return await slack_handler.handle(request)
        except Exception as e:
            logger.error(f"Error handling Slack event: {e}")
            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )

    @app.get("/slack/health", tags=["Slack"])
    async def slack_health_check():
        """Slack bot health check"""
        return {
            "status": "healthy",
            "bot_name": "Jira Bot",
            "message": "Slack bot is running",
        }

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "jira-bot-api"}

    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "Jira Bot API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "docs": "/docs",
                "ask_query": "/ask-query",
                "jira_query": "/test-jira-query",
                "slack_events": "/slack/events",
                "slack_health": "/slack/health",
            },
        }

    return app


# Create app instance
app = create_app()
