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
from datetime import datetime
from fastapi.responses import PlainTextResponse
from fastapi import Request

from .utilities.utils import Utils

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize Slack app
slack_app = App(
    token=os.getenv("SLACK_BOT_TOKEN"),
    signing_secret=os.getenv("SLACK_SIGNING_SECRET"),
)


def call_jira_api(query, channel_id=None, message_id=None):
    """Helper function to call Jira API with optional channel and message ID"""
    try:
        api_endpoint = f"{os.getenv('APP_BACKEND_URL')}ask-query"

        # Prepare payload with channel_id and message_id if provided
        payload = {"query": query}
        if channel_id:
            payload["channel_id"] = channel_id
        if message_id:
            payload["message_id"] = message_id

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
        channel_id = event.get("channel")
        message_id = event.get("ts")  # User's message timestamp
        user_query = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

        # Log channel information
        logger.info(f"Message from channel: {channel_id}, message_id: {message_id}")

        if not user_query:
            say(
                "Please provide a Jira request after mentioning me. Example: `@jirabot create a bug in AI project for login issues`"
            )
            return

        # Pass channel_id and message_id to the API call
        answer = call_jira_api(user_query, channel_id, message_id)

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
        message_id = message.get("ts")  # User's message timestamp

        # Log channel information
        logger.info(
            f"Direct message from channel: {channel_id}, message_id: {message_id}"
        )

        if not user_query:
            say(
                "Please provide a Jira request. Example: `create a task in AI project for user authentication`"
            )
            return

        if user_query.lower().startswith("ask"):
            return

        # Pass channel_id and message_id to the API call
        answer = call_jira_api(user_query, channel_id, message_id)
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
        message_id = message.get("ts")  # User's message timestamp

        # Log channel information
        logger.info(
            f"Create message from channel: {channel_id}, message_id: {message_id}"
        )

        # Pass channel_id and message_id to the API call
        answer = call_jira_api(user_query, channel_id, message_id)
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
        Simplified endpoint that handles both JSON (API calls) and form data (Slack slash commands)
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

                    # Extract channel_id and message_id if provided in JSON payload
                    channel_id = body.get("channel_id")
                    message_id = body.get("message_id")

                    if channel_id:
                        logger.info(
                            f"API request from channel: {channel_id}, message_id: {message_id}"
                        )

                    user_query = UserQuery(query=body["query"])

                    # Pass both channel_id and message_id to the service
                    result = await self.jira_service.process_query(
                        user_query, channel_id, message_id
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
                    channel_id = form_data.get("channel_id", "")
                    response_url = form_data.get("response_url", "")  # NEED THIS!

                    logger.info(
                        f"Slack - Command: {command}, Text: {text}, Channel: {channel_id}"
                    )

                    if command != "/jiratest":
                        return {
                            "response_type": "ephemeral",
                            "text": f"Unknown command: {command}",
                        }

                    if not text:
                        return {
                            "response_type": "ephemeral",
                            "text": "Please provide a request. Example: `/jiratest create a bug in AI project`",
                        }

                    # START BACKGROUND PROCESSING - DON'T WAIT!
                    background_tasks.add_task(
                        self.process_slash_command_async, text, channel_id, response_url
                    )

                    # RETURN IMMEDIATELY - NO PROCESSING HERE!
                    return {
                        "response_type": "in_channel",
                        "text": "⏳ Processing ...",
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

    async def process_slash_command_async(
        self, text: str, channel_id: str, response_url: str
    ):
        """Background task to process slash command and update message"""
        try:
            logger.info(f"Processing slash command in background: {text}")

            # Process the query
            user_query = UserQuery(query=text)
            result = await self.jira_service.process_query(user_query, None, None)

            # Post result to channel and get real timestamp
            try:
                from slack_sdk import WebClient

                client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

                response = client.chat_postMessage(
                    channel=channel_id,
                    text=result.get("data", ""),
                    unfurl_links=True,
                    unfurl_media=True,
                )

                if response["ok"]:
                    real_timestamp = response["ts"]
                    logger.info(f"Posted result with real timestamp: {real_timestamp}")

                    # Process again with real timestamp for tracking
                    await self.jira_service.process_query(
                        user_query, channel_id, real_timestamp
                    )

                # Replace the "Processing..." message
                final_response = {
                    "response_type": "in_channel",
                    "replace_original": True,
                    "text": "",  # Empty to remove the processing message
                }

                requests.post(response_url, json=final_response, timeout=10)

            except Exception as post_error:
                logger.error(f"Error posting result: {post_error}")
                # Fallback: update via response_url
                final_response = {
                    "response_type": "in_channel",
                    "replace_original": True,
                    "text": result.get("data", ""),
                }
                requests.post(response_url, json=final_response, timeout=30)

        except Exception as e:
            logger.error(f"Error in background slash command processing: {e}")
            try:
                error_response = {
                    "response_type": "in_channel",
                    "replace_original": True,
                    "text": "Sorry, something went wrong while processing your request.",
                }
                requests.post(response_url, json=error_response, timeout=10)
            except Exception as send_error:
                logger.error(f"Failed to send error response: {send_error}")


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

    @app.post("/jira-webhook", tags=["Slack"])
    async def jira_webhook(request: Request):
        LAST_STATUS = {}
        RECENT = {}

        payload = await request.json()
        issue = payload.get("issue", {})
        fields = issue.get("fields", {})
        issue_key = issue.get("key", "UNKNOWN")
        ts = int(payload.get("timestamp") or 0)

        # Deduplication
        last_ts = RECENT.get(issue_key, 0)
        if ts and last_ts and ts <= last_ts:
            return PlainTextResponse("", status_code=200)
        RECENT[issue_key] = ts

        # Find status change
        items = payload.get("changelog", {}).get("items", [])
        status_item = next((i for i in items if i.get("field") == "status"), None)
        if not status_item:
            return PlainTextResponse("", status_code=200)

        from_status = (status_item.get("fromString") or "").strip()
        to_status = (status_item.get("toString") or "").strip()
        if not from_status or not to_status or from_status == to_status:
            return PlainTextResponse("", status_code=200)

        # Ignore automation actors
        actor = payload.get("user", {})
        actor_name = actor.get("displayName", "")
        if "automation" in actor_name.lower():
            return PlainTextResponse("", status_code=200)

        # Current status
        status_obj = fields.get("status", {})
        current_status = (status_obj.get("name") or "").strip().lower()
        current_cat = (status_obj.get("statusCategory", {}).get("key") or "").lower()
        is_now_done = (current_status == "done") or (current_cat == "done")

        # Dedup status transitions
        last_status = LAST_STATUS.get(issue_key)
        if to_status.lower() == last_status:
            print(f"[SKIP] {issue_key}: Status did not change (still {to_status})")
            return PlainTextResponse("", status_code=200)

        print(f"Status changed: {from_status} → {to_status}")
        print("issue_key:", issue_key, "\n\n")
        Utils.postStatusMsgToSlack(issue_key, to_status)

        LAST_STATUS[issue_key] = to_status.lower()
        return PlainTextResponse("", status_code=200)

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
