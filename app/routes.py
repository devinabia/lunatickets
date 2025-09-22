# -*- coding: utf-8 -*-
import os
import re
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple

from dotenv import load_dotenv

from fastapi import FastAPI, APIRouter, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse

from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_sdk import WebClient

# Your existing imports
from .views import JiraService
from .schemas import UserQuery
from .utilities.utils import Utils

# -----------------------------------------------------------------------------
# Setup & Init
# -----------------------------------------------------------------------------
load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
BACKEND_URL = os.getenv("APP_BACKEND_URL")

slack_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
slack_handler = SlackRequestHandler(slack_app)
slack_client = WebClient(token=SLACK_BOT_TOKEN)

# Resolve bot user id (for exact mention token detection)
BOT_USER_ID = os.getenv("SLACK_BOT_USER_ID")
if not BOT_USER_ID:
    try:
        whoami = slack_client.auth_test()
        BOT_USER_ID = whoami.get("user_id")
    except Exception:
        BOT_USER_ID = None

BOT_MENTION = f"<@{BOT_USER_ID}>" if BOT_USER_ID else None


def _now() -> datetime:
    return datetime.utcnow()


# -----------------------------------------------------------------------------
# User mention/ID → name helpers (used in call_jira_api and mention paths)
# -----------------------------------------------------------------------------
# Matches: <@U123>, <@U123|label>
MENTION_TOKEN_RE = re.compile(r"<@([A-Z0-9]+)(?:\|[^>]+)?>")
# Matches bare IDs like U123ABC456 (U/W prefix, 6–20 chars after), with loose boundaries
BARE_ID_RE = re.compile(r"(?<![A-Z0-9])(U|W)[A-Z0-9]{6,20}(?![A-Z0-9])")

# user_id -> {"name": str, "ts": datetime}
_USER_CACHE: Dict[str, Dict[str, Any]] = {}
USER_CACHE_TTL_MIN = int(os.getenv("USER_CACHE_TTL_MIN", "240"))


def _user_cache_get(uid: str) -> Optional[str]:
    info = _USER_CACHE.get(uid)
    if not info:
        return None
    if (_now() - info["ts"]) > timedelta(minutes=USER_CACHE_TTL_MIN):
        _USER_CACHE.pop(uid, None)
        return None
    return info["name"]


def _user_cache_set(uid: str, name: str) -> None:
    _USER_CACHE[uid] = {"name": name, "ts": _now()}


def get_user_display_name(uid: str) -> str:
    """Resolve a Slack user ID to a friendly name (display → real name → uid), with debug logging."""
    if not uid:
        return uid
    cached = _user_cache_get(uid)
    if cached:
        logger.debug(f"[users_info] cache hit {uid} -> {cached}")
        return cached
    try:
        resp = slack_client.users_info(user=uid)
        if not resp.get("ok"):
            err = resp.get("error", "unknown_error")
            logger.warning(
                f"[users_info] failed for {uid}: {err} (check users:read scope & reinstall)"
            )
            return uid
        user = resp.get("user", {}) or {}
        profile = user.get("profile", {}) or {}
        display = (
            profile.get("display_name_normalized")
            or profile.get("display_name")
            or user.get("real_name")
            or uid
        )
        name = display.strip() if isinstance(display, str) and display.strip() else uid
        _user_cache_set(uid, name)
        logger.info(f"[users_info] resolved {uid} -> {name}")
        return name
    except Exception as e:
        logger.exception(f"[users_info] exception for {uid}: {e}")
        return uid


def contains_bot_mention(text: Optional[str]) -> bool:
    return bool(BOT_MENTION and text and BOT_MENTION in text)


def replace_mention_tokens_with_names(text: str) -> str:
    """Replace <@U…> (and <@U…|label>) tokens with the user's display name; keep bot token intact."""
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        uid = m.group(1)
        if BOT_USER_ID and uid == BOT_USER_ID:
            return m.group(0)  # preserve bot token; we strip it elsewhere if needed
        return get_user_display_name(uid)

    return MENTION_TOKEN_RE.sub(_repl, text)


def replace_bare_ids_with_names(text: str) -> str:
    """Replace bare Slack IDs (U…/W…) with display names (skip bot)."""
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        uid = m.group(0)
        if BOT_USER_ID and uid == BOT_USER_ID:
            return uid
        return get_user_display_name(uid)

    return BARE_ID_RE.sub(_repl, text)


def expand_all_user_refs_to_names(text: str) -> str:
    """
    Convert both mention tokens and bare IDs into names.
    Order matters: handle tokens first (to avoid turning <@U..> into a partial).
    """
    if not text:
        return text
    t = replace_mention_tokens_with_names(text)
    t = replace_bare_ids_with_names(t)
    return t


# -----------------------------------------------------------------------------
# Thread session state
# -----------------------------------------------------------------------------
# thread_ts -> {"invoker": user_id, "last_activity": datetime}
ACTIVE_THREADS: Dict[str, Dict[str, Any]] = {}
THREAD_TTL_MIN = int(os.getenv("THREAD_TTL_MIN", "120"))  # default: 2 hours


def _purge_expired_threads() -> None:
    cutoff = _now() - timedelta(minutes=THREAD_TTL_MIN)
    for t, meta in list(ACTIVE_THREADS.items()):
        if meta.get("last_activity", cutoff) < cutoff:
            ACTIVE_THREADS.pop(t, None)


def _mark_thread_active(
    thread_ts: Optional[str], invoker_user: Optional[str] = None
) -> None:
    if not thread_ts:
        return
    meta = ACTIVE_THREADS.get(thread_ts, {})
    if invoker_user and not meta.get("invoker"):
        meta["invoker"] = invoker_user
    meta["last_activity"] = _now()
    ACTIVE_THREADS[thread_ts] = meta


def _is_bot_message(event: Dict[str, Any]) -> bool:
    subtype = event.get("subtype")
    if event.get("bot_id"):
        return True
    if subtype in {
        "bot_message",
        "message_changed",
        "message_deleted",
        "channel_join",
        "channel_leave",
    }:
        return True
    return False


# -----------------------------------------------------------------------------
# One-time reply de-dup (per Slack message)
# -----------------------------------------------------------------------------
_HANDLED: Dict[Tuple[str, str], datetime] = {}
DEDUP_TTL_SEC = int(os.getenv("DEDUP_TTL_SEC", "120"))


def _already_handled(channel_id: Optional[str], message_ts: Optional[str]) -> bool:
    if not channel_id or not message_ts:
        return False
    key = (channel_id, message_ts)

    cutoff = _now() - timedelta(seconds=DEDUP_TTL_SEC)
    for k, t in list(_HANDLED.items()):
        if t < cutoff:
            _HANDLED.pop(k, None)

    if key in _HANDLED:
        return True
    _HANDLED[key] = _now()
    return False


# -----------------------------------------------------------------------------
# Intent detection
# -----------------------------------------------------------------------------
COMMAND_PATTERNS = [
    r"^\s*(create|make|open)\b.*\b(ticket|issue|bug|story|task|epic)\b",
    r"^\s*(update|change|modify|edit)\b.*\b(issue|ticket|bug|story|task|epic)\b",
    r"^\s*(assign|reassign)\b.*\b(to|@)?\b",
    r"^\s*(set|change)\b.*\b(priority|status|estimate|assignee)\b",
    r"^\s*(link|unlink|relate)\b.*\b(issue|ticket)\b",
    r"^\s*(comment|add comment)\b",
    r"^\s*(transition|move)\b.*\b(status)\b",
    r"^\s*(close|resolve|reopen)\b.*\b(issue|ticket|bug|story|task|epic)\b",
    r"\b[A-Z]{2,}-\d+\b",
]
COMMAND_RE = re.compile("|".join(f"(?:{p})" for p in COMMAND_PATTERNS), re.IGNORECASE)
PHRASE_PREV_DISCUSSION = re.compile(
    r"\bbased on (the )?previous (discussion|conversation|messages)\b", re.IGNORECASE
)


def looks_like_bot_intent(text: str) -> bool:
    if not text:
        return False
    if contains_bot_mention(text):
        return True
    if COMMAND_RE.search(text):
        return True
    if re.match(
        r"^\s*(create|update|assign|close|resolve|set|change|comment|transition|move)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    return False


# -----------------------------------------------------------------------------
# Jira API helper (NORMALIZES Slack user refs → names before sending)
# -----------------------------------------------------------------------------
def call_jira_api(
    query: str, channel_id: Optional[str] = None, message_id: Optional[str] = None
) -> str:
    try:
        # Normalize refs → names (covers <@U…>, <@U…|label>, bare U…/W…)
        normalized_query = expand_all_user_refs_to_names(query)

        # Also strip bot mention token if it slipped through here
        if BOT_MENTION and BOT_MENTION in normalized_query:
            normalized_query = normalized_query.replace(BOT_MENTION, "").strip()

        # Debug for verification
        logger.info(f"RAW: {query}")
        logger.info(f"NORM: {normalized_query}")

        api_endpoint = f"{BACKEND_URL}ask-query"
        payload: Dict[str, Any] = {"query": normalized_query}
        if channel_id:
            payload["channel_id"] = channel_id
        if message_id:
            payload["message_id"] = message_id

        resp = requests.post(
            api_endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if resp.status_code != 200:
            return f"API Error: {resp.status_code}"
        data = resp.json()
        if data.get("success"):
            return data.get("data") or "No response from Jira"
        return f"Error: {data.get('message', 'Unknown error occurred')}"
    except Exception:
        logger.exception("Error calling Jira API")
        return "Something went wrong. Please try again."


# -----------------------------------------------------------------------------
# Utility: show & remove "Processing…" placeholder
# -----------------------------------------------------------------------------
def post_processing_notice(channel_id: str, thread_ts: str) -> Optional[str]:
    try:
        res = slack_client.chat_postMessage(
            channel=channel_id,
            text="⏳ Processing …",
            thread_ts=thread_ts,
            unfurl_links=False,
            unfurl_media=False,
        )
        if res.get("ok"):
            return res["ts"]
    except Exception as e:
        logger.warning(f"post_processing_notice failed: {e}")
    return None


def delete_message(channel_id: str, ts: str) -> None:
    try:
        slack_client.chat_delete(channel=channel_id, ts=ts)
    except Exception as e:
        logger.warning(f"delete_message failed: {e}")


# -----------------------------------------------------------------------------
# Slack Listeners
# -----------------------------------------------------------------------------
# FIX 404: Ack message_changed / message_deleted etc. so Bolt doesn't return 404.
@slack_app.event({"type": "message", "subtype": "message_changed"})
def _ack_message_changed(body, logger):
    logger.info("Ignoring message_changed event")


@slack_app.event({"type": "message", "subtype": "message_deleted"})
def _ack_message_deleted(body, logger):
    logger.info("Ignoring message_deleted event")


# (Add other noisy subtypes here if needed)
# @slack_app.event({"type": "message", "subtype": "channel_join"})
# def _ack_join(body, logger): logger.info("Ignoring channel_join")


@slack_app.event("app_mention")
def handle_app_mention(event, say):
    """
    ANY explicit mention of the bot:
      - Reply exactly once (de-dup).
      - Start/continue the thread session.
      - Convert other user mentions/IDs to names before calling Jira.
      - Show 'Processing …' placeholder, then delete it before final reply.
    """
    try:
        logger.info(f"[app_mention] {event}")
        channel_id = event.get("channel")
        message_id = event.get("ts")
        user_id = event.get("user")
        text = (event.get("text", "") or "").strip()

        if _already_handled(channel_id, message_id):
            return

        thread_id = event.get("thread_ts") or message_id
        _mark_thread_active(thread_id, invoker_user=user_id)

        cleaned = replace_mention_tokens_with_names(text)
        if BOT_MENTION:
            cleaned = cleaned.replace(BOT_MENTION, "")
        cleaned = cleaned.strip()

        if not cleaned:
            say(
                {
                    "text": "Please provide a Jira request after mentioning me. Example:\n`@jirabot create a bug in AI project for login issues`",
                    "thread_ts": thread_id,
                }
            )
            return

        # placeholder
        ph_ts = post_processing_notice(channel_id, thread_id)

        answer = call_jira_api(cleaned, channel_id, message_id)

        # remove placeholder then answer
        if ph_ts:
            delete_message(channel_id, ph_ts)

        say(
            {
                "text": answer,
                "unfurl_links": True,
                "unfurl_media": True,
                "thread_ts": thread_id,
            }
        )

    except Exception:
        logger.exception("Error in handle_app_mention")
        say(
            {
                "text": "Something went wrong. Please try again.",
                "thread_ts": event.get("thread_ts") or event.get("ts"),
            }
        )


@slack_app.message("")  # catch-all
def handle_messages(message, say):
    """
    Handles:
      - DMs (no mention needed)
      - Thread replies in active threads (no mention needed)
      - Mentions as a BACKSTOP when Slack doesn't deliver app_mention (de-dup prevents double)
    Avoids:
      - bot/system messages
      - non-thread channel chatter
    """
    try:
        if _is_bot_message(message):
            return

        channel_type = message.get("channel_type")  # "im" for DMs
        channel_id = message.get("channel")
        message_id = message.get("ts")
        user_id = message.get("user")
        text = (message.get("text") or "").strip()
        thread_ts = message.get("thread_ts")

        if _already_handled(channel_id, message_id):
            return

        _purge_expired_threads()

        # 1) DMs
        if channel_type == "im":
            if not text:
                say(
                    "Please provide a Jira request. Example: `create a task in AI project for user authentication`"
                )
                return

            thread_id = thread_ts or message_id
            ph_ts = post_processing_notice(channel_id, thread_id)

            answer = call_jira_api(text, channel_id, message_id)

            if ph_ts:
                delete_message(channel_id, ph_ts)

            say(
                {
                    "text": answer,
                    "thread_ts": thread_id,
                    "unfurl_links": True,
                    "unfurl_media": True,
                }
            )
            return

        # 2) Mentions (fallback if app_mention didn't fire)
        if contains_bot_mention(text):
            thread_id = thread_ts or message_id
            _mark_thread_active(thread_id, invoker_user=user_id)

            cleaned = replace_mention_tokens_with_names(text)
            if BOT_MENTION:
                cleaned = cleaned.replace(BOT_MENTION, "")
            cleaned = cleaned.strip()

            if not cleaned:
                say({"text": "How can I help with Jira?", "thread_ts": thread_id})
                return

            ph_ts = post_processing_notice(channel_id, thread_id)

            answer = call_jira_api(cleaned, channel_id, message_id)

            if ph_ts:
                delete_message(channel_id, ph_ts)

            say(
                {
                    "text": answer,
                    "unfurl_links": True,
                    "unfurl_media": True,
                    "thread_ts": thread_id,
                }
            )
            return

        # 3) Channel/group threads: only if the thread is active
        if thread_ts:
            session_id = thread_ts
            if session_id in ACTIVE_THREADS:
                invoker = ACTIVE_THREADS[session_id].get("invoker")
                is_invoker = user_id == invoker

                should_reply = bool(text) and (
                    is_invoker or looks_like_bot_intent(text)
                )
                if should_reply:
                    _mark_thread_active(session_id)  # refresh TTL

                    context_text = ""
                    if PHRASE_PREV_DISCUSSION.search(text):
                        try:
                            replies = slack_client.conversations_replies(
                                channel=channel_id, ts=session_id, limit=20
                            )
                            msgs = replies.get("messages", [])
                            recent_human = []
                            for m in reversed(msgs[-8:]):
                                if (
                                    not m.get("bot_id")
                                    and "text" in m
                                    and m.get("user")
                                ):
                                    recent_human.append(m["text"])
                                if len(recent_human) >= 4:
                                    break
                            if recent_human:
                                context_text = "\n\nContext:\n" + "\n".join(
                                    f"- {t}" for t in recent_human
                                )
                        except Exception:
                            pass

                    full_query = text + context_text

                    ph_ts = post_processing_notice(channel_id, session_id)

                    answer = call_jira_api(full_query, channel_id, message_id)

                    if ph_ts:
                        delete_message(channel_id, ph_ts)

                    say(
                        {
                            "text": answer,
                            "unfurl_links": True,
                            "unfurl_media": True,
                            "thread_ts": session_id,
                        }
                    )
                return  # handled or ignored in thread

        # 4) Non-thread channel messages: ignore
        return

    except Exception:
        logger.exception("Error in handle_messages")
        ts = message.get("thread_ts") or message.get("ts")
        say(
            {"text": "Something went wrong. Please try again.", "thread_ts": ts}
            if ts
            else {"text": "Something went wrong. Please try again."}
        )


# Optional convenience: 'create ...' in DM or active thread
@slack_app.message(re.compile(r"^\s*create\b", re.IGNORECASE))
def handle_create_shortcut(message, say):
    try:
        if _is_bot_message(message):
            return
        channel_id = message.get("channel")
        message_id = message.get("ts")
        text = (message.get("text") or "").strip()
        thread_ts = message.get("thread_ts")
        if _already_handled(channel_id, message_id):
            return
        if message.get("channel_type") == "im" or (
            thread_ts and thread_ts in ACTIVE_THREADS
        ):
            ph_ts = post_processing_notice(channel_id, thread_ts or message_id)
            answer = call_jira_api(text, channel_id, message_id)
            if ph_ts:
                delete_message(channel_id, ph_ts)
            say(
                {
                    "text": answer,
                    "unfurl_links": True,
                    "unfurl_media": True,
                    "thread_ts": thread_ts or message_id,
                }
            )
    except Exception:
        logger.exception("Error in handle_create_shortcut")


# -----------------------------------------------------------------------------
# FastAPI app + routes
# -----------------------------------------------------------------------------
class BotRouter:
    def __init__(self):
        self.jira_service = JiraService()
        self.router = APIRouter()
        self.setup_routes()

    def setup_routes(self):
        self.router.add_api_route("/ask-query", self.handle_ask_query, methods=["POST"])
        self.router.add_api_route(
            "/test-jira-query", self.jira_service.process_query, methods=["POST"]
        )

    async def handle_jira_query(self, request: Request):
        try:
            if (
                "application/json"
                in (request.headers.get("content-type") or "").lower()
            ):
                body = await request.json()
                user_query = body.get("query")
                if not user_query:
                    raise HTTPException(status_code=422, detail="Missing 'query' field")
                result = self.jira_service.process_query(user_query)
                return result
            raise HTTPException(
                status_code=415, detail="Unsupported content type. Use application/json"
            )
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error in handle_jira_query")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def handle_ask_query(
        self, request: Request, background_tasks: BackgroundTasks
    ):
        try:
            ctype = (request.headers.get("content-type") or "").lower()
            logger.info(f"Received request with content-type: {ctype}")

            # JSON (programmatic API)
            if "application/json" in ctype:
                body = await request.json()
                if "query" not in body:
                    raise HTTPException(status_code=422, detail="Missing 'query' field")

                channel_id = body.get("channel_id")
                message_id = body.get("message_id")
                if channel_id:
                    logger.info(
                        f"API request from channel: {channel_id}, message_id: {message_id}"
                    )

                user_query = UserQuery(query=body["query"])
                result = await self.jira_service.process_query(
                    user_query, channel_id, message_id
                )
                return result

            # Slack slash command
            if "application/x-www-form-urlencoded" in ctype:
                form = await request.form()
                command = form.get("command", "")
                text = (form.get("text") or "").strip()
                channel_id = form.get("channel_id", "")
                response_url = form.get("response_url", "")

                logger.info(
                    f"Slack - Command: {command}, Text: {text}, Channel: {channel_id}"
                )

                if command != "/jira":
                    return {
                        "response_type": "ephemeral",
                        "text": f"Unknown command: {command}",
                    }
                if not text:
                    return {
                        "response_type": "ephemeral",
                        "text": "Please provide a request. Example: `/jira create a bug in AI project`",
                    }

                background_tasks.add_task(
                    self.process_slash_command_async, text, channel_id, response_url
                )
                return {"response_type": "in_channel", "text": "⏳ Processing ..."}

            raise HTTPException(
                status_code=415, detail=f"Unsupported content type: {ctype}"
            )

        except HTTPException:
            raise
        except Exception:
            logger.exception("General error in handle_ask_query")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def process_slash_command_async(
        self, text: str, channel_id: str, response_url: str
    ):
        try:
            user_query = UserQuery(query=text)
            result = await self.jira_service.process_query(
                user_query, channel_id, "SLASH_COMMAND"
            )
            final_response = {
                "response_type": "in_channel",
                "replace_original": True,
                "text": result.get("data", ""),
            }
            requests.post(response_url, json=final_response, timeout=10)
        except Exception:
            logger.exception("Error in process_slash_command_async")
            try:
                requests.post(
                    response_url,
                    json={
                        "response_type": "in_channel",
                        "replace_original": True,
                        "text": "Sorry, something went wrong while processing your request.",
                    },
                    timeout=10,
                )
            except Exception:
                logger.exception("Failed to send error response")


def create_app():
    app = FastAPI(
        title="Jira Bot API",
        description="API for Jira operations with Slack integration",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    bot_router = BotRouter()
    app.include_router(bot_router.router, tags=["Bot"])

    @app.post("/jira-webhook", tags=["Slack"])
    async def jira_webhook(request: Request):
        LAST_STATUS: Dict[str, str] = {}
        RECENT: Dict[str, int] = {}

        payload = await request.json()
        issue = payload.get("issue", {})
        fields = issue.get("fields", {})
        issue_key = issue.get("key", "UNKNOWN")
        ts = int(payload.get("timestamp") or 0)

        # Dedup
        last_ts = RECENT.get(issue_key, 0)
        if ts and last_ts and ts <= last_ts:
            return PlainTextResponse("", status_code=200)
        RECENT[issue_key] = ts

        # Status change?
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

        status_obj = fields.get("status", {})
        current_status = (status_obj.get("name") or "").strip().lower()
        current_cat = (status_obj.get("statusCategory", {}).get("key") or "").lower()
        _ = (current_status == "done") or (current_cat == "done")

        last_status = LAST_STATUS.get(issue_key)
        if to_status.lower() == last_status:
            return PlainTextResponse("", status_code=200)

        Utils.postStatusMsgToSlack(issue_key, to_status)
        LAST_STATUS[issue_key] = to_status.lower()
        return PlainTextResponse("", status_code=200)

    @app.post("/slack/events", tags=["Slack"])
    async def slack_events(request: Request):
        try:
            logger.info(f"Slack event received: {request.method} {request.url}")
            return await slack_handler.handle(request)
        except Exception:
            logger.exception("Error handling Slack event")
            return JSONResponse(
                status_code=500, content={"error": "Internal server error"}
            )

    @app.get("/slack/health", tags=["Slack"])
    async def slack_health_check():
        return {
            "status": "healthy",
            "bot_name": "Jira Bot",
            "message": "Slack bot is running",
        }

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "jira-bot-api"}

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
