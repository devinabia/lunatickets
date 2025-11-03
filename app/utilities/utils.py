import os, re, asyncio
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from slack_sdk import WebClient
from dotenv import load_dotenv
from slack_sdk.errors import SlackApiError
from openai import OpenAI
from qdrant_client import QdrantClient


load_dotenv()
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
logger = logging.getLogger(__name__)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


if os.getenv("ORG") == "INABIA":
    JIRA_ACCOUNTS = {
        "default": {
            "name": "Default Account",
            "base_url": os.getenv("JIRA_BASE_URL"),
            "email": os.getenv("JIRA_EMAIL"),
            "token": os.getenv("JIRA_TOKEN"),
            "project_key": os.getenv("Default_Project"),
            "description": "Main Jira workspace for general projects",
        },
        "ark": {
            "name": "ARK Account",
            "base_url": os.getenv("JIRA_ARK_BASE_URL"),
            "email": os.getenv("JIRA_ARK_EMAIL"),
            "token": os.getenv("JIRA_ARK_TOKEN"),
            "project_key": os.getenv("JIRA_ARK_PROJECT", "AB"),
            "description": "ARK-specific Jira workspace",
        },
    }
else:
    JIRA_ACCOUNTS = {
        "default": {
            "name": "Default Account",
            "base_url": os.getenv("JIRA_BASE_URL"),
            "email": os.getenv("JIRA_EMAIL"),
            "token": os.getenv("JIRA_TOKEN"),
            "project_key": os.getenv("Default_Project"),
            "description": "Main Jira workspace for general projects",
        }
    }


class Utils:
    """Pure utility class - all intelligence handled by LangGraph agent."""

    def __init__(self, base_url, email, token, session):
        self.base_url = base_url
        self.email = email
        self.token = token
        self.session = session
        self.current_account = "default"  # NEW: Track active account

    @staticmethod
    def get_account_config(account_key: str = "default"):
        """Get account configuration safely with fallback to default"""
        return JIRA_ACCOUNTS.get(account_key, JIRA_ACCOUNTS["default"])

    def switch_account(self, account_key: str) -> bool:
        """
        Switch to a different Jira account dynamically.
        Updates base_url, credentials, and session authentication.

        Args:
            account_key: Account identifier ('default', 'ark', etc.)

        Returns:
            bool: True if switch successful, False otherwise

        Examples:
            utils.switch_account("ark")  # Switch to Ark account
            utils.switch_account("default")  # Switch back to default
        """
        try:
            # Import here to avoid circular imports

            config = Utils.get_account_config(account_key)

            # Validate config
            if not all(k in config for k in ["base_url", "email", "token"]):
                logger.error(f"Invalid config for account: {account_key}")
                return False

            # Update instance variables
            self.current_account = account_key
            self.base_url = config["base_url"]
            self.email = config["email"]
            self.token = config["token"]

            # Update session authentication
            self.session.auth = (self.email, self.token)

            logger.info(
                f"✅ Switched to Jira account: '{account_key}' ({config.get('name', 'Unknown')})"
            )
            logger.info(f"   Base URL: {self.base_url}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to switch account to '{account_key}': {e}")
            return False

    def get_current_account(self) -> str:
        """Get currently active account key"""
        return self.current_account

    def get_current_account_info(self) -> dict:
        """Get full info about currently active account"""
        try:

            return Utils.get_account_config(self.current_account)
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {
                "name": "Unknown",
                "base_url": self.base_url,
                "current_account": self.current_account,
            }

    # ========================================
    # SLACK INTEGRATION (RAW DATA ONLY)
    # ========================================

    def extract_chat(self, channel_id, message_id=None):
        """Extract Slack channel messages (and threads) in readable format.
        If message_id belongs to a thread, include a 'Current Thread' section.
        """
        if not channel_id:
            return ""

        today = datetime.now().astimezone()
        day_start = today.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=2)
        oldest = day_start.timestamp()
        latest = min(day_end.timestamp(), today.timestamp())

        messages = []
        cursor = None

        try:
            # ---- Fetch all messages (and thread replies) ----
            while True:
                resp = client.conversations_history(
                    channel=channel_id,
                    oldest=str(oldest),
                    latest=str(latest),
                    inclusive=True,
                    limit=200,
                    cursor=cursor,
                )
                batch = resp.get("messages", [])
                if not batch:
                    break

                for m in batch:
                    if m.get("subtype") == "message_deleted":
                        continue
                    if m.get("thread_ts") and m["thread_ts"] != m["ts"]:
                        # skip replies in history — they'll be fetched via conversations_replies
                        continue

                    m["_depth"] = 0
                    m["_parent_ts"] = m["ts"]
                    messages.append(m)

                    # Fetch replies
                    if (m.get("reply_count") or 0) > 0:
                        rcur = None
                        while True:
                            try:
                                r = client.conversations_replies(
                                    channel=channel_id,
                                    ts=m["ts"],
                                    limit=200,
                                    cursor=rcur,
                                )
                            except SlackApiError as e:
                                if e.response.status_code == 429:
                                    time.sleep(
                                        int(e.response.headers.get("Retry-After", "2"))
                                    )
                                    continue
                                break

                            r_msgs = r.get("messages", [])
                            for rep in r_msgs:
                                if rep.get("ts") == m["ts"]:
                                    continue
                                rep["_depth"] = 1
                                rep["_parent_ts"] = m["ts"]
                                messages.append(rep)

                            rcur = r.get("response_metadata", {}).get("next_cursor")
                            if not rcur:
                                break

                cursor = resp.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

            # ---- Sort messages like Slack (parent → replies) ----
            messages.sort(
                key=lambda msg: (
                    float(msg.get("_parent_ts") or msg["ts"]),
                    msg["_depth"],
                    float(msg["ts"]),
                )
            )

            if not messages:
                return ""

            # ---- Helpers ----
            user_cache = {}

            def get_user_name(user_id, bot_profile=None):
                if bot_profile and bot_profile.get("name"):
                    return bot_profile["name"]
                if not user_id or user_id == "bot":
                    return "Bot"
                if user_id in user_cache:
                    return user_cache[user_id]
                try:
                    ui = client.users_info(user=user_id)
                    if ui.get("ok"):
                        name = (
                            ui["user"].get("display_name")
                            or ui["user"].get("real_name")
                            or user_id
                        )
                        user_cache[user_id] = name
                        return name
                except Exception:
                    pass
                user_cache[user_id] = user_id
                return user_id

            def clean_text(t: str) -> str:
                if not t:
                    return ""
                t = re.sub(
                    r"<@([A-Z0-9]+)>", lambda m: f"@{get_user_name(m.group(1))}", t
                )
                t = re.sub(r"<#([A-Z0-9]+)\|([^>]+)>", r"#\2", t)
                t = re.sub(r"<(https?://[^>|]+)(\|[^>]+)?>", r"\1", t)
                return t.strip()

            def extract_text(msg: dict) -> str:
                t = (msg.get("text") or "").strip()
                if not t and msg.get("blocks"):
                    parts = []
                    for b in msg["blocks"]:
                        if b.get("type") == "rich_text":
                            for el in b.get("elements", []):
                                if el.get("type") == "rich_text_section":
                                    for s in el.get("elements", []):
                                        if s.get("type") == "text" and s.get("text"):
                                            parts.append(s["text"])
                        elif b.get("type") == "section" and b.get("text", {}).get(
                            "text"
                        ):
                            parts.append(b["text"]["text"])
                    t = " ".join(parts)
                if not t and msg.get("files"):
                    t = " ".join([f.get("name") for f in msg["files"] if f.get("name")])
                return t

            def format_messages(msgs):
                lines = []
                for msg in msgs:
                    txt = clean_text(extract_text(msg))
                    if not txt:
                        continue
                    ts = datetime.fromtimestamp(
                        float(msg["ts"]), tz=timezone.utc
                    ).strftime("%H:%M")
                    uname = get_user_name(msg.get("user"), msg.get("bot_profile"))
                    indent = "    " * msg["_depth"]
                    arrow = "↳ " if msg["_depth"] else ""
                    lines.append(f"{indent}{ts} {uname}: {arrow}{txt}")
                return lines

            def ts_equal(a, b):
                try:
                    return abs(float(a) - float(b)) < 0.001
                except Exception:
                    return False

            # ---- Determine current thread if message_id provided ----
            current_thread_ts = None
            if message_id:
                for msg in messages:
                    if ts_equal(msg["ts"], message_id):
                        current_thread_ts = msg.get("thread_ts") or msg["ts"]
                        break

                # Fetch missing replies if not already in memory
                if current_thread_ts and not any(
                    m.get("_parent_ts") == current_thread_ts for m in messages
                ):
                    try:
                        r = client.conversations_replies(
                            channel=channel_id, ts=current_thread_ts, limit=200
                        )
                        for rep in r.get("messages", []):
                            if rep["ts"] != current_thread_ts:
                                rep["_depth"] = 1
                                rep["_parent_ts"] = current_thread_ts
                                messages.append(rep)
                        # Re-sort after adding new messages
                        messages.sort(
                            key=lambda msg: (
                                float(msg.get("_parent_ts") or msg["ts"]),
                                msg["_depth"],
                                float(msg["ts"]),
                            )
                        )
                    except Exception as e:
                        logger.warning(f"Failed to fetch thread replies: {e}")

            # ---- Separate messages into Previous and Current threads ----
            if current_thread_ts:
                # Separate current thread messages from other messages
                current_thread_msgs = []
                previous_msgs = []

                for msg in messages:
                    parent_ts = msg.get("_parent_ts") or msg["ts"]
                    if parent_ts == current_thread_ts:
                        current_thread_msgs.append(msg)
                    else:
                        previous_msgs.append(msg)

                # Build output with both sections
                chat_output = ""

                # Add Previous Chat section (excluding current thread)
                # if previous_msgs:
                #     chat_output = "--- 📜 Previous Chat ---\n"
                #     formatted_prev = format_messages(previous_msgs)
                #     # Limit to last 150 messages for previous chat to save space
                #     chat_output += "\n".join(formatted_prev[-150:])

                # Add Current Thread section
                if current_thread_msgs:
                    if chat_output:
                        chat_output += "\n\n"
                    chat_output += "--- 💬 Current Thread ---\n"
                    chat_output += "\n".join(format_messages(current_thread_msgs))
            else:
                # No specific thread selected, show all as general chat
                chat_output = "--- 📜 Channel Messages ---\n"
                formatted_lines = format_messages(messages)
                chat_output += "\n".join(formatted_lines[-200:])

            return chat_output

        except Exception as e:
            logger.error(f"Error extracting chat: {e}", exc_info=True)
            return ""

    def format_for_slack(self, text: str) -> str:
        """Format response text for Slack markdown with clickable issue keys."""
        if not text:
            return text

        import re

        # Convert [text](url) to <url|text>
        text = re.sub(r"\[([^\]]+?)\]\(([^)]+?)\)", r"<\2|\1>", text)

        # Make issue keys clickable (pure formatting, not intelligence)
        issue_key_pattern = r"(?<!browse/)(?<!browse%2F)\b([A-Z]+[-_]\d+)\b(?![^<]*>)"

        def make_issue_clickable(match):

            url_dict = {
                os.getenv("JIRA_BASE_URL"): os.getenv("JIRA_DOMAIN_URL"),
                os.getenv("JIRA_ARK_BASE_URL"): os.getenv("JIRA_AKR_DOMAIN_URL"),
            }
            get_ticket_url = url_dict.get(self.base_url)
            # Build the ticket URL
            ticket_url = f"{get_ticket_url.rstrip('/')}/browse/{issue_key}"


            issue_key = match.group(1)
            issue_url = f"{self.base_url.rstrip('/')}/browse/{issue_key}"
            return f"<{issue_url}|{issue_key}>"

        text = re.sub(issue_key_pattern, make_issue_clickable, text)

        # Convert markdown bold for Slack
        text = re.sub(r"(?<!\*)\*\*(?!\*)([^*]+?)(?<!\*)\*\*(?!\*)", r"*\1*", text)

        # Clean up any broken nested links
        text = re.sub(r"<([^>]*)<([^>]*)>", r"<\2>", text)
        text = re.sub(r"%7C", "|", text)

        return text

    # ========================================
    # JIRA API UTILITIES (PURE UTILITIES)
    # ========================================

    def text_to_adf(self, text: str) -> dict:
        """Convert plain text to Atlassian Document Format."""
        if not text:
            return {"type": "doc", "version": 1, "content": [{"type": "paragraph"}]}

        parts = text.split("\n")
        content = []
        for i, chunk in enumerate(parts):
            if chunk:
                content.append({"type": "text", "text": chunk})
            if i < len(parts) - 1:
                content.append({"type": "hardBreak"})

        return {
            "type": "doc",
            "version": 1,
            "content": [{"type": "paragraph", "content": content}],
        }

    def resolve_project_key(self, name_or_key: str) -> str:
        """
        Return project key by searching both Jira projects and Confluence spaces.

        If name_or_key is empty or None, returns the default project from environment.

        Search Priority:
        1. Use default project if name_or_key is empty
        2. Exact Jira project key match
        3. Exact Jira project name match
        4. Partial Jira project name match
        5. Search Confluence spaces and find linked Jira project

        Examples:
            - "" -> "AI" (uses default project)
            - None -> "AI" (uses default project)
            - "DATA" -> "DATA" (exact key match)
            - "Customers" -> "CUST" (exact name match)
            - "data squad" -> "DATA" (found via Confluence space)
        """
        # ========================================
        # STEP 0: Handle empty/None with default
        # ========================================
        if not name_or_key or not name_or_key.strip():
            current_config = Utils.get_account_config(self.current_account)
            default_project = current_config.get(
                "project_key", os.getenv("Default_Project")
            )

            if not default_project or not default_project.strip():
                raise RuntimeError(
                    "No project specified and current account has no default project. "
                    f"Current account: {self.current_account}"
                )
            logger.info(
                f"✅ Using current account's default project: {default_project} (account: {self.current_account})"
            )
            return default_project.strip()

        try:
            # ========================================
            # STEP 1: Search in Jira Projects
            # ========================================
            r = self.session.get(
                f"{self.base_url}/rest/api/3/project/search", timeout=20
            )
            r.raise_for_status()
            projects = r.json().get("values", [])

            if not projects:
                logger.warning("No Jira projects found")

            search_term = name_or_key.strip().lower()

            # Priority 1: Exact key match (case-insensitive)
            for p in projects:
                if p["key"].lower() == search_term:
                    logger.info(f"✅ Exact key match: '{name_or_key}' -> {p['key']}")
                    return p["key"]

            # Priority 2: Exact name match (case-insensitive)
            for p in projects:
                if p["name"].lower() == search_term:
                    logger.info(
                        f"✅ Exact name match: '{name_or_key}' -> {p['key']} ({p['name']})"
                    )
                    return p["key"]

            # Priority 3: Partial name match
            for p in projects:
                if search_term in p["name"].lower() or p["name"].lower() in search_term:
                    logger.info(
                        f"✅ Partial name match: '{name_or_key}' -> {p['key']} ({p['name']})"
                    )
                    return p["key"]

            logger.info(
                f"⚠️ No match in Jira projects for '{name_or_key}', searching Confluence spaces..."
            )

            # ========================================
            # STEP 2: Search in Confluence Spaces
            # ========================================
            confluence_match = self._search_confluence_spaces_for_project(name_or_key)

            if confluence_match:
                logger.info(
                    f"✅ Found via Confluence space: '{name_or_key}' -> {confluence_match['project_key']} (Space: {confluence_match['space_name']})"
                )
                return confluence_match["project_key"]

            # ========================================
            # STEP 3: No match found anywhere
            # ========================================
            available_projects = [f"{p['key']} ({p['name']})" for p in projects[:10]]
            error_msg = (
                f"Project or space '{name_or_key}' not found.\n"
                f"Available Jira projects: {', '.join(available_projects)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        except Exception as e:
            logger.error(f"Error fetching projects from Jira: {e}")
            raise RuntimeError(f"Failed to resolve project: {str(e)}")

    def _search_confluence_spaces_for_project(self, space_name: str) -> dict:
        """
        Search Confluence spaces and return the linked Jira project key.

        Args:
            space_name: Confluence space name (e.g., "data squad")

        Returns:
            dict: {"project_key": "DATA", "space_name": "Data Squad"} or None
        """
        try:
            from jira import JIRA

            # Initialize Jira client
            jira_client = JIRA(
                server=self.base_url, basic_auth=(self.email, self.token)
            )

            # Get all projects to check their keys
            all_projects = jira_client.projects()

            search_term = space_name.strip().lower()

            # Try to match space name with project key or name patterns
            for project in all_projects:
                # Check if the space name matches project key
                if project.key.lower() == search_term:
                    return {
                        "project_key": project.key,
                        "space_name": space_name,
                        "match_type": "key_match",
                    }

                # Check if space name is similar to project name
                project_name_lower = project.name.lower()

                # Exact match
                if project_name_lower == search_term:
                    return {
                        "project_key": project.key,
                        "space_name": project.name,
                        "match_type": "name_match",
                    }

                # Check if space name words are in project key
                # Example: "data squad" -> words: ["data", "squad"] -> check if "DATA" contains them
                space_words = search_term.split()
                project_key_lower = project.key.lower()

                # If first word of space matches project key
                if space_words and space_words[0] in project_key_lower:
                    return {
                        "project_key": project.key,
                        "space_name": space_name,
                        "match_type": "fuzzy_match",
                    }

            # Alternative: Try using Confluence API if available
            # This requires Confluence access which you might have
            try:
                confluence_result = self._search_confluence_api(space_name)
                if confluence_result:
                    return confluence_result
            except Exception as conf_error:
                logger.warning(f"Confluence API search failed: {conf_error}")

            return None

        except ImportError:
            logger.warning("jira library not available for space search")
            return None
        except Exception as e:
            logger.warning(f"Error searching Confluence spaces: {e}")
            return None

    def _search_confluence_api(self, space_name: str) -> dict:
        """
        Search Confluence spaces using REST API and find linked Jira projects.

        This method attempts to find the Jira project linked to a Confluence space.
        """
        try:
            # Get Confluence spaces
            confluence_url = self.base_url.replace("/jira", "/wiki")

            # Search for spaces
            r = self.session.get(
                f"{confluence_url}/rest/api/space", params={"limit": 100}, timeout=20
            )

            if r.status_code != 200:
                logger.warning(f"Could not access Confluence API: {r.status_code}")
                return None

            spaces = r.json().get("results", [])
            search_term = space_name.strip().lower()

            for space in spaces:
                space_key = space.get("key", "").lower()
                space_display_name = space.get("name", "").lower()

                # Check if space name matches
                if search_term == space_display_name or search_term == space_key:
                    # Try to find linked Jira project
                    # Usually the space key matches the project key
                    space_key_upper = space.get("key", "").upper()

                    # Verify this project exists in Jira
                    try:
                        verify_response = self.session.get(
                            f"{self.base_url}/rest/api/3/project/{space_key_upper}",
                            timeout=10,
                        )
                        if verify_response.status_code == 200:
                            return {
                                "project_key": space_key_upper,
                                "space_name": space.get("name"),
                                "match_type": "confluence_api",
                            }
                    except Exception:
                        pass

            return None

        except Exception as e:
            logger.warning(f"Confluence API search error: {e}")
            return None

    def get_project_users(self, project_key: str, max_results: int = 50) -> list:
        """Get list of users assignable to issues in a project."""
        try:
            r = self.session.get(
                f"{self.base_url}/rest/api/3/user/assignable/search",
                params={"project": project_key, "maxResults": max_results},
                timeout=20,
            )
            r.raise_for_status()
            users = r.json()

            formatted_users = []
            for user in users:
                user_data = {
                    "accountId": user["accountId"],
                    "displayName": user["displayName"],
                    "emailAddress": user.get("emailAddress", ""),
                    "active": user.get("active", True),
                }
                formatted_users.append(user_data)

            logger.info(
                f"Found {len(formatted_users)} assignable users for project {project_key}"
            )
            return formatted_users
        except Exception as e:
            logger.error(f"Error getting project users for {project_key}: {e}")
            return []

    def find_user_by_name_or_email(self, project_key: str, query: str) -> dict:
        """Find a user in the project by display name or email address."""
        try:
            users = self.get_project_users(project_key)
            query_lower = query.lower().strip()

            # Try exact matches first
            for user in users:
                if (
                    user["emailAddress"].lower() == query_lower
                    or user["displayName"].lower() == query_lower
                ):
                    return user

            # Try partial matches
            for user in users:
                if (
                    query_lower in user["displayName"].lower()
                    or query_lower in user["emailAddress"].lower()
                ):
                    return user

            logger.warning(
                f"No user found for query '{query}' in project {project_key}"
            )
            return None
        except Exception as e:
            logger.error(f"Error finding user '{query}' in project {project_key}: {e}")
            return None

    def get_user_suggestions_text(self, project_key: str, limit: int = 10) -> str:
        """Get formatted text list of available users for assignment suggestions."""
        try:
            users = self.get_project_users(project_key, max_results=limit)
            if not users:
                return f"No assignable users found for project {project_key}"

            suggestion_lines = [f"Available users in {project_key} project:"]
            for i, user in enumerate(users[:limit], 1):
                email_part = (
                    f" ({user['emailAddress']})" if user["emailAddress"] else ""
                )
                suggestion_lines.append(f"{i}. {user['displayName']}{email_part}")

            if len(users) > limit:
                suggestion_lines.append(f"... and {len(users) - limit} more users")

            return "\n".join(suggestion_lines)
        except Exception as e:
            logger.error(f"Error generating user suggestions for {project_key}: {e}")
            return f"Could not retrieve user list for project {project_key}"

    def smart_assign_user(self, project_key: str, assignee_input: str) -> dict:
        """Intelligently assign a user based on input."""
        try:
            if not assignee_input or assignee_input.strip().lower() in [
                "",
                "unassigned",
                "none",
            ]:
                return {
                    "success": True,
                    "accountId": None,
                    "displayName": "Unassigned",
                    "message": "Ticket will be left unassigned",
                }

            user = self.find_user_by_name_or_email(project_key, assignee_input)
            if user:
                return {
                    "success": True,
                    "accountId": user["accountId"],
                    "displayName": user["displayName"],
                    "message": f"Found user: {user['displayName']}",
                }
            else:
                suggestions = self.get_user_suggestions_text(project_key)
                return {
                    "success": False,
                    "accountId": None,
                    "displayName": None,
                    "message": f"User '{assignee_input}' not found in project {project_key}",
                    "suggestions": suggestions,
                }
        except Exception as e:
            logger.error(f"Error in smart_assign_user: {e}")
            return {
                "success": False,
                "accountId": None,
                "displayName": None,
                "message": f"Error finding user: {str(e)}",
            }

    def get_priority_id_by_name(self, name: str) -> str:
        """Get priority ID by name."""
        r = self.session.get(f"{self.base_url}/rest/api/3/priority", timeout=20)
        r.raise_for_status()
        for pr in r.json():
            if pr["name"].lower() == name.lower():
                return pr["id"]
        raise RuntimeError(f"Priority '{name}' not found.")

    def normalize_issue_type(self, project_key: str, issue_type_name: str) -> str:
        """Normalize issue type name to match Jira's expectations."""
        default_issue_type = "Story"
        if not issue_type_name:
            return default_issue_type

        valid_types = self.get_valid_issue_types(project_key)
        logger.info(
            f"Valid issue types for project {project_key}: {list(valid_types.values())}"
        )

        # Try exact match first
        if issue_type_name in valid_types.values():
            return issue_type_name

        # Try case-insensitive match
        normalized = issue_type_name.lower()
        if normalized in valid_types:
            return valid_types[normalized]

        # Common mappings
        type_aliases = {
            "bug": ["story", "task"],
            "task": ["task"],
            "story": ["story"],
            "epic": ["epic"],
            "subtask": ["subtask"],
            "sub-task": ["subtask"],
        }

        if normalized in type_aliases:
            for alias_target in type_aliases[normalized]:
                if alias_target in valid_types:
                    logger.info(
                        f"Mapping '{issue_type_name}' to '{valid_types[alias_target]}'"
                    )
                    return valid_types[alias_target]

        # Use default if nothing matches
        if valid_types:
            logger.warning(
                f"Issue type '{issue_type_name}' not found. Available types: {list(valid_types.values())}"
            )
            if "task" in valid_types:
                logger.info("Using default 'Task' type")
                return valid_types["task"]
            else:
                first_type = list(valid_types.values())[0]
                logger.info(f"Using first available type: '{first_type}'")
                return first_type

        return default_issue_type

    def get_valid_issue_types(self, project_key: str) -> dict:
        """Get valid issue types for the project."""
        try:
            r = self.session.get(
                f"{self.base_url}/rest/api/3/issue/createmeta",
                params={"projectKeys": project_key, "expand": "projects.issuetypes"},
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            projects = data.get("projects") or []

            if projects:
                issue_types = projects[0].get("issuetypes", [])
                type_mapping = {}
                for issue_type in issue_types:
                    name = issue_type["name"]
                    type_mapping[name.lower()] = name
                return type_mapping
            return {}
        except Exception as e:
            logger.error(f"Error getting issue types: {e}")
            return {}

    def get_create_fields(self, project_key: str, issue_type_name: str) -> set:
        """Get fields allowed on create screen."""
        try:
            r = self.session.get(
                f"{self.base_url}/rest/api/3/issue/createmeta",
                params={
                    "projectKeys": project_key,
                    "issuetypeNames": issue_type_name,
                    "expand": "projects.issuetypes.fields",
                },
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()
            projects = data.get("projects") or []

            if not projects:
                logger.warning(f"No projects found for key {project_key}")
                return set()

            issue_types = projects[0].get("issuetypes") or []
            if not issue_types:
                logger.warning(f"No issue types found for {issue_type_name}")
                return set()

            fields = issue_types[0].get("fields") or {}
            field_keys = set(fields.keys())
            logger.info(f"Available fields for {issue_type_name}: {field_keys}")
            return field_keys
        except Exception as e:
            logger.error(f"Error getting create fields: {e}")
            return {
                "project",
                "summary",
                "description",
                "issuetype",
                "assignee",
                "priority",
                "reporter",
            }

    def get_board_info(self, project_key: str) -> dict:
        """Get board information for the project."""
        try:
            r = self.session.get(
                f"{self.base_url}/rest/agile/1.0/board",
                params={"projectKeyOrId": project_key},
                timeout=20,
            )
            r.raise_for_status()
            boards = r.json().get("values", [])

            if boards:
                board = boards[0]
                board_id = board["id"]
                r = self.session.get(
                    f"{self.base_url}/rest/agile/1.0/board/{board_id}/configuration",
                    timeout=20,
                )
                r.raise_for_status()
                config = r.json()

                return {
                    "board_id": board_id,
                    "board_name": board["name"],
                    "board_type": board["type"],
                    "filter": config.get("filter", {}),
                }
            return None
        except Exception as e:
            logger.error(f"Error getting board info: {e}")
            return None

    def get_issue(
        self,
        issue_key: str,
        fields: str = "summary,description,priority,assignee,reporter,status",
    ):
        """Get issue details."""
        r = self.session.get(
            f"{self.base_url}/rest/api/3/issue/{issue_key}",
            params={"fields": fields},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()

    def update_description(self, issue_key: str, description_text: str) -> None:
        """Update issue description."""
        if not description_text or not description_text.strip():
            return

        try:
            url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
            payload = {
                "fields": {
                    "description": {
                        "type": "doc",
                        "version": 1,
                        "content": [
                            {
                                "type": "paragraph",
                                "content": [{"type": "text", "text": description_text}],
                            }
                        ],
                    }
                }
            }

            r = self.session.put(url, json=payload, timeout=20)
            if r.ok:
                logger.info(f"Description updated for {issue_key}")
            else:
                logger.error(f"Description update failed: {r.status_code} - {r.text}")
                raise RuntimeError(f"Update failed: {r.text}")
        except Exception as e:
            logger.error(f"Error updating description: {e}")
            raise

    # ========================================
    # NEW: STORY POINTS AND EPIC SUPPORT
    # ========================================

    def get_story_points_field_id(self, project_key: str) -> str:
        """Find the story points custom field ID for the project."""
        try:
            # Common story points field IDs
            common_story_fields = [
                "customfield_10016",  # Most common
                "customfield_10002",
                "customfield_10004",
                "customfield_10008",
                "customfield_10020",
            ]

            # Try to get field mapping from create meta
            r = self.session.get(
                f"{self.base_url}/rest/api/3/issue/createmeta",
                params={
                    "projectKeys": project_key,
                    "expand": "projects.issuetypes.fields",
                },
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()

            for project in data.get("projects", []):
                for issue_type in project.get("issuetypes", []):
                    fields = issue_type.get("fields", {})
                    for field_id, field_info in fields.items():
                        field_name = field_info.get("name", "").lower()
                        if "story" in field_name and "point" in field_name:
                            logger.info(f"Found story points field: {field_id}")
                            return field_id

            # Fallback to most common
            logger.info("Using default story points field: customfield_10016")
            return "customfield_10016"

        except Exception as e:
            logger.warning(f"Error finding story points field: {e}")
            return "customfield_10016"

    def get_epic_link_field_id(self, project_key: str) -> str:
        """Find the epic link custom field ID for the project."""
        try:
            # Common epic link field IDs
            common_epic_fields = [
                "customfield_10014",  # Most common
                "customfield_10006",
                "customfield_10008",
                "customfield_10010",
            ]

            # Try to detect from create meta
            r = self.session.get(
                f"{self.base_url}/rest/api/3/issue/createmeta",
                params={
                    "projectKeys": project_key,
                    "expand": "projects.issuetypes.fields",
                },
                timeout=20,
            )
            r.raise_for_status()
            data = r.json()

            for project in data.get("projects", []):
                for issue_type in project.get("issuetypes", []):
                    fields = issue_type.get("fields", {})
                    for field_id, field_info in fields.items():
                        field_name = field_info.get("name", "").lower()
                        if "epic" in field_name and "link" in field_name:
                            logger.info(f"Found epic link field: {field_id}")
                            return field_id

            # Fallback to most common
            logger.info("Using default epic link field: customfield_10014")
            return "customfield_10014"

        except Exception as e:
            logger.warning(f"Error finding epic link field: {e}")
            return "customfield_10014"

    def get_project_epics_implementation(self, project_key: str) -> dict:
        """Get epics using the jira Python library for better reliability."""
        try:
            logger.info(
                f"=== Fetching epics for project {project_key} using jira library ==="
            )

            # Import the jira library
            from jira import JIRA

            # Initialize the Jira client using the same credentials
            jira_client = JIRA(
                server=self.base_url, basic_auth=(self.email, self.token)
            )

            # Search for epics using JQL - same as your working script
            jql_query = (
                f"project = {project_key} AND issuetype = Epic ORDER BY created DESC"
            )

            # Search for epics (limit to 50 for performance)
            epic_issues = jira_client.search_issues(jql_query, maxResults=50)

            logger.info(f"Found {len(epic_issues)} epics in project {project_key}")

            if epic_issues:
                epics = []
                for epic in epic_issues:
                    epic_info = {
                        "key": epic.key,
                        "summary": epic.fields.summary,
                        "status": epic.fields.status.name,
                    }
                    epics.append(epic_info)
                    logger.info(f"Epic found: {epic.key} - {epic.fields.summary}")

                # Format for display
                formatted_list = []
                for epic in epics:
                    formatted_list.append(
                        f"• {epic['key']}: {epic['summary']} ({epic['status']})"
                    )

                return {
                    "success": True,
                    "project": project_key,
                    "epics": epics,
                    "formatted_list": "\n".join(formatted_list),
                    "message": f"Found {len(epics)} epics in {project_key}",
                    "method_used": "jira_library",
                }
            else:
                logger.info(f"No epics found in project {project_key}")
                return {
                    "success": True,
                    "project": project_key,
                    "epics": [],
                    "formatted_list": f"No epics found in project {project_key}.\n\nTo create an epic, go to your Jira project and create a new issue with type 'Epic'.",
                    "message": f"No epics found in {project_key}. You may need to create some epics first.",
                    "method_used": "jira_library",
                }

        except ImportError as import_error:
            logger.error(f"jira library not installed: {import_error}")
            return {
                "success": False,
                "project": project_key,
                "epics": [],
                "error": "jira library not installed",
                "message": f"Epic discovery requires the 'jira' Python library. Please install it with: pip install jira",
                "formatted_list": "Epic discovery unavailable - jira library missing.",
            }

        except Exception as e:
            logger.error(f"Error fetching epics using jira library: {e}")

            # Provide helpful fallback message
            return {
                "success": True,  # Set to True so ticket creation can still proceed
                "project": project_key,
                "epics": [],
                "error": str(e),
                "formatted_list": f"Epic auto-discovery temporarily unavailable.\n\nYou can manually specify epic keys from your project (e.g., {project_key}-123).\n\nTo find epics, go to your Jira project → Issues and filter by Epic issue type.",
                "message": f"Cannot automatically list epics right now. You can specify epic keys manually.",
                "method_used": "fallback",
            }

    # ========================================
    # SPRINT UTILITIES
    # ========================================

    def _get_board_id_for_project(self, project_key: str) -> int:
        """Find a board for this project (prefer scrum)."""
        if not project_key:
            current_config = Utils.get_account_config(self.current_account)
            project_key = current_config.get(
                "project_key", os.getenv("Default_Project")
            )
            logger.info(
                f"📌 Using current account's project for board lookup: {project_key}"
            )

        url = f"{self.base_url}/rest/agile/1.0/board"
        r = self.session.get(
            url, params={"projectKeyOrId": project_key, "maxResults": 50}, timeout=30
        )
        r.raise_for_status()
        boards = r.json().get("values", [])
        if not boards:
            return None
        # Prefer scrum boards (they have sprints)
        scrum = [b for b in boards if b.get("type") == "scrum"]
        return scrum[0]["id"] if scrum else boards[0]["id"]

    def _get_sprint_id_by_name(self, board_id: int, sprint_name: str) -> int:
        """Find a sprint by exact name on the given board."""
        start_at = 0
        while True:
            r = self.session.get(
                f"{self.base_url}/rest/agile/1.0/board/{board_id}/sprint",
                params={
                    "startAt": start_at,
                    "maxResults": 50,
                    "state": "active,future,closed",
                },
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            for s in data.get("values", []):
                if s.get("name", "").strip().lower() == sprint_name.strip().lower():
                    return s["id"]
            if data.get("isLast") or not data.get("values"):
                break
            start_at += len(data.get("values", []))
        return None

    def _add_issue_to_sprint(self, sprint_id: int, issue_key: str) -> None:
        """Move an issue into the given sprint."""
        r = self.session.post(
            f"{self.base_url}/rest/agile/1.0/sprint/{sprint_id}/issue",
            json={"issues": [issue_key]},
            timeout=30,
        )
        r.raise_for_status()

    def get_all_sprints_for_project(self, project_key: str) -> str:
        """Get formatted list of sprints for a project."""
        try:
            if not project_key:
                current_config = Utils.get_account_config(self.current_account)
                project_key = current_config.get(
                    "project_key", os.getenv("Default_Project")
                )
                logger.info(
                    f"📌 Using current account's project for sprint list: {project_key}"
                )

            board_id = self._get_board_id_for_project(project_key)
            if not board_id:
                return f"No board found for project {project_key}. Use 'backlog' for no sprint."

            all_sprints = []
            start_at = 0

            while True:
                r = self.session.get(
                    f"{self.base_url}/rest/agile/1.0/board/{board_id}/sprint",
                    params={
                        "startAt": start_at,
                        "maxResults": 50,
                        "state": "active,future,closed",
                    },
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()

                for sprint in data.get("values", []):
                    all_sprints.append(
                        {
                            "id": sprint["id"],
                            "name": sprint["name"],
                            "state": sprint["state"],
                            "startDate": sprint.get("startDate"),
                            "endDate": sprint.get("endDate"),
                        }
                    )

                if data.get("isLast") or not data.get("values"):
                    break
                start_at += len(data.get("values", []))

            # Sort by ID descending (latest first)
            all_sprints.sort(key=lambda x: x["id"], reverse=True)

            # Format for display
            sprint_options = [
                f"Available sprints for {project_key}:",
                "- backlog (no specific sprint)",
            ]
            for sprint in all_sprints[:10]:  # Limit to 10 latest sprints
                status_marker = ""
                if sprint["state"] == "active":
                    status_marker = " (ONGOING)"
                elif sprint["state"] == "future":
                    status_marker = " (upcoming)"
                sprint_options.append(f"- {sprint['name']}{status_marker}")

            return "\n".join(sprint_options)
        except Exception as e:
            logger.error(f"Error getting sprints for {project_key}: {e}")
            return f"Could not retrieve sprints for {project_key}. Use 'backlog' for no sprint."

    def get_default_sprint_for_project(self, project_key: str) -> dict:
        """Get the default sprint for a project (the immediate next sprint after ongoing)."""
        try:
            if not project_key:
                current_config = Utils.get_account_config(self.current_account)
                project_key = current_config.get(
                    "project_key", os.getenv("Default_Project")
                )
                logger.info(
                    f"📌 Using current account's project for default sprint: {project_key}"
                )

            sprint_info = self.get_all_sprints_for_project(project_key)

            if "No board found" in sprint_info:
                return {
                    "has_default": False,
                    "sprint_name": None,
                    "ask_user": True,
                    "sprint_list": sprint_info,
                }

            board_id = self._get_board_id_for_project(project_key)
            if not board_id:
                return {
                    "has_default": False,
                    "sprint_name": None,
                    "ask_user": True,
                    "sprint_list": sprint_info,
                }

            # Get all active and future sprints
            r = self.session.get(
                f"{self.base_url}/rest/agile/1.0/board/{board_id}/sprint",
                params={"state": "active,future", "maxResults": 50},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()

            future_sprints = []
            for sprint in data.get("values", []):
                if sprint["state"] == "future":
                    future_sprints.append(sprint)

            if not future_sprints:
                return {
                    "has_default": False,
                    "sprint_name": None,
                    "ask_user": True,
                    "sprint_list": sprint_info,
                }

            # Sort future sprints by week number extracted from name
            def extract_week_number(sprint_name):
                import re

                try:
                    match = re.search(r"W(\d+)", sprint_name)
                    if match:
                        return int(match.group(1))
                    numbers = re.findall(r"\d+", sprint_name)
                    if numbers:
                        return int(numbers[0])
                    return 0
                except:
                    return 0

            future_sprints.sort(key=lambda x: extract_week_number(x["name"]))
            next_sprint = future_sprints[0]

            logger.info(
                f"Using immediate next sprint as default: {next_sprint['name']}"
            )
            return {
                "has_default": True,
                "sprint_name": next_sprint["name"],
                "ask_user": False,
                "sprint_list": sprint_info,
            }

        except Exception as e:
            logger.error(f"Error getting default sprint for {project_key}: {e}")
            return {
                "has_default": False,
                "sprint_name": None,
                "ask_user": True,
                "sprint_list": f"Could not get sprints for {project_key}. Use 'backlog'.",
            }

    def get_sprint_list_implementation(self, project_name_or_key: str) -> dict:
        """Get list of sprints for a project."""
        try:
            # ✅ FIX: Default to current account's project if empty
            if not project_name_or_key:
                current_config = Utils.get_account_config(self.current_account)
                project_name_or_key = current_config.get(
                    "project_key", os.getenv("Default_Project")
                )
                logger.info(
                    f"📌 Using current account's project for sprints: {project_name_or_key}"
                )

            project_key = self.resolve_project_key(project_name_or_key)
            sprint_list = self.get_all_sprints_for_project(project_key)

            return {
                "success": True,
                "project_key": project_key,
                "sprint_list": sprint_list,
            }
        except Exception as e:
            logger.error(f"Error getting sprint list: {e}")
            return {
                "success": False,
                "error": str(e),
                "sprint_list": f"Could not get sprints for {project_name_or_key}. Use 'backlog'.",
            }

    def get_project_from_issue_implementation(self, issue_key: str) -> dict:
        """Helper tool to get the project key from an issue key."""
        try:
            issue_data = self.get_issue(issue_key, "project")
            project_key = issue_data["fields"]["project"]["key"]

            return {"success": True, "issue_key": issue_key, "project_key": project_key}
        except Exception as e:
            logger.error(f"Error getting project for issue {issue_key}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Could not find project for issue {issue_key}",
            }

    # ========================================
    # CRUD OPERATIONS (UPDATED WITH STORY POINTS AND EPIC)
    # ========================================

    def create_issue_implementation(
        self,
        project_name_or_key: str = str(os.getenv("Default_Project")),
        summary: str = "",
        description_text: str = "",
        assignee_email: str = "",
        priority_name: str = None,
        reporter_email: str = None,
        issue_type_name: str = None,
        sprint_name: str = None,
        story_points: int = None,
        epic_key: str = None,
        slack_username: str = None,  # NEW PARAMETER
        channel_id: str = None,  # NEW
        message_id: str = None,  # NEW
        force_update_description_after_create: bool = True,
    ) -> dict:
        """Create a new Jira ticket/issue with validation and automatic reporter matching."""
        if not project_name_or_key or not project_name_or_key.strip():
            current_config = Utils.get_account_config(self.current_account)
            project_name_or_key = current_config.get(
                "project_key", os.getenv("Default_Project")
            )
            logger.info(f"📌 Using current account's project: {project_name_or_key}")

        logger.info(
            f"Creating issue with context - assignee: {assignee_email}, summary: {summary}, "
            f"story_points: {story_points}, epic: {epic_key}, slack_user: {slack_username}"
        )

        if issue_type_name is None:
            issue_type_name = "Story"

        try:
            project_key = self.resolve_project_key(project_name_or_key)
            logger.info(
                f"✅ Resolved project '{project_name_or_key}' -> '{project_key}'"
            )
        except Exception as resolve_error:
            logger.error(
                f"❌ Failed to resolve project '{project_name_or_key}': {resolve_error}"
            )
            return {
                "success": False,
                "error": str(resolve_error),
                "message": f"Could not find project '{project_name_or_key}'. Please check the project name or key. {str(resolve_error)}",
            }

        # Generate default summary if empty
        if not summary or summary.strip() == "":
            summary = "New ticket created via AI assistant"

        # Generate default description if empty
        if not description_text or description_text.strip() == "":
            description_text = "This ticket was created through the AI assistant and needs further details to be added."

        if channel_id and message_id:
            slack_link = self.build_slack_thread_link(channel_id, message_id)
            if slack_link:
                description_text = self.append_slack_link_to_description(
                    description_text, slack_link
                )
                logger.info(f"✅ Added Slack thread link to description: {slack_link}")
            else:
                logger.warning("⚠️ Failed to generate Slack thread link")

        try:

            # Handle sprint selection with user fallback
            if not sprint_name:
                sprint_info = self.get_default_sprint_for_project(project_key)

                if sprint_info["has_default"]:
                    sprint_name = sprint_info["sprint_name"]
                    logger.info(f"Using default upcoming sprint: {sprint_name}")
                else:
                    logger.info(
                        f"No upcoming sprint available for {project_key}, asking user"
                    )
                    return {
                        "success": False,
                        "needs_sprint_selection": True,
                        "message": f"Which sprint should I add this ticket to? Here are the available options:\n\n{sprint_info['sprint_list']}\n\nPlease specify the exact sprint name or 'backlog'.",
                        "project": project_key,
                        "assignee": assignee_email,
                        "sprint_options": sprint_info["sprint_list"],
                    }

            normalized_issue_type = self.normalize_issue_type(
                project_key, issue_type_name
            )
            logger.info(
                f"Original issue type: '{issue_type_name}' -> Normalized: '{normalized_issue_type}'"
            )

            allowed = self.get_create_fields(project_key, normalized_issue_type)
            board_info = self.get_board_info(project_key)

            fields = {
                "project": {"key": project_key},
                "summary": summary.strip().strip("'\""),
                "issuetype": {"name": normalized_issue_type},
            }

            # Set description
            if "description" in allowed and description_text:
                fields["description"] = self.text_to_adf(description_text)

            # Smart assignee handling
            assignment_info = {
                "assigned": False,
                "assignee_name": "Unassigned",
                "suggestions": None,
            }

            if "assignee" in allowed and assignee_email and assignee_email.strip():
                assignment_result = self.smart_assign_user(
                    project_key, assignee_email.strip()
                )

                if assignment_result["success"]:
                    if assignment_result["accountId"]:
                        fields["assignee"] = {"id": assignment_result["accountId"]}
                        assignment_info["assigned"] = True
                        assignment_info["assignee_name"] = assignment_result[
                            "displayName"
                        ]
                        logger.info(
                            f"Successfully assigned to: {assignment_result['displayName']}"
                        )
                else:
                    logger.warning(
                        f"Could not assign to '{assignee_email}': {assignment_result['message']}"
                    )
                    assignment_info["suggestions"] = assignment_result.get(
                        "suggestions", ""
                    )

            # NEW: Smart reporter handling based on Slack username
            reporter_info = {
                "set": False,
                "name": "Default (API Token User)",
                "match_type": None,
                "error": None,
            }

            if "reporter" in allowed:
                # Priority 1: Try to match Slack username to Jira user
                if slack_username and slack_username.strip():
                    logger.info(
                        f"Attempting to match Slack user '{slack_username}' to Jira reporter"
                    )
                    reporter_match = self.find_reporter_by_slack_username(
                        project_key, slack_username.strip()
                    )

                    if reporter_match["success"]:
                        fields["reporter"] = {"id": reporter_match["accountId"]}
                        reporter_info["set"] = True
                        reporter_info["name"] = reporter_match["displayName"]
                        reporter_info["match_type"] = reporter_match.get(
                            "matchType", "unknown"
                        )
                        reporter_info["confidence"] = reporter_match.get("score", 0)
                        logger.info(
                            f"✅ Set reporter to '{reporter_match['displayName']}' based on "
                            f"Slack user '{slack_username}' ({reporter_match.get('matchType', 'unknown')} match)"
                        )
                    else:
                        logger.warning(
                            f"⚠️ Could not match Slack user '{slack_username}' to Jira user: "
                            f"{reporter_match['message']}"
                        )
                        reporter_info["error"] = reporter_match["message"]

                # Priority 2: Fallback to reporter_email if provided and no slack match
                if (
                    not reporter_info["set"]
                    and reporter_email
                    and reporter_email.strip()
                ):
                    try:
                        reporter_id = self.get_account_id(reporter_email.strip())
                        fields["reporter"] = {"id": reporter_id}
                        reporter_info["set"] = True
                        reporter_info["name"] = reporter_email.strip()
                        reporter_info["match_type"] = "email"
                        logger.info(
                            f"Set reporter to {reporter_email} via email parameter"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Could not set reporter '{reporter_email}': {e}"
                        )
                        reporter_info["error"] = str(e)

            # Set priority
            if priority_name and "priority" in allowed:
                try:
                    pr_id = self.get_priority_id_by_name(priority_name)
                    fields["priority"] = {"id": pr_id}
                    logger.info(f"Set priority to: {priority_name}")
                except Exception as e:
                    logger.warning(f"Could not set priority '{priority_name}': {e}")

            # Set story points if provided
            if story_points and normalized_issue_type in ["Story", "Task"]:
                try:
                    story_points_field = self.get_story_points_field_id(project_key)
                    if story_points_field in allowed:
                        fields[story_points_field] = story_points
                        logger.info(f"Setting story points to: {story_points}")
                except Exception as e:
                    logger.warning(f"Could not set story points '{story_points}': {e}")

            # Set epic link if provided
            if epic_key and normalized_issue_type != "Epic":
                try:
                    epic_link_field = self.get_epic_link_field_id(project_key)
                    if epic_link_field in allowed:
                        fields[epic_link_field] = epic_key
                        logger.info(f"Linking to epic: {epic_key}")
                except Exception as e:
                    logger.warning(f"Could not link to epic '{epic_key}': {e}")

            # Add labels if supported
            if "labels" in allowed:
                fields["labels"] = ["created-by-luna"]

            logger.info(f"Creating issue with fields: {list(fields.keys())}")

            # Create the issue
            create_url = f"{self.base_url}/rest/api/3/issue"
            resp = self.session.post(create_url, json={"fields": fields}, timeout=30)

            if not resp.ok:
                logger.error(f"Create failed: {resp.status_code} - {resp.text}")
                raise RuntimeError(
                    f"Jira create failed {resp.status_code}: {resp.text}"
                )

            created = resp.json()
            issue_key = created.get("key")
            logger.info(f"✅ Successfully created issue: {issue_key}")

            # Handle sprint assignment
            sprint_status = "Backlog"
            if sprint_name and sprint_name.strip():
                try:
                    board_id = self._get_board_id_for_project(project_key)
                    if not board_id:
                        raise RuntimeError(f"No board found for project {project_key}")
                    sprint_id = self._get_sprint_id_by_name(
                        board_id, sprint_name.strip()
                    )
                    if not sprint_id:
                        raise RuntimeError(
                            f"{sprint_name.strip()} not found on this board"
                        )
                    self._add_issue_to_sprint(sprint_id, issue_key)
                    sprint_status = sprint_name.strip()
                    logger.info(
                        f"Issue {issue_key} moved to sprint {sprint_name.strip()} (id={sprint_id})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to move issue to sprint {sprint_name}: {e}")
                    sprint_status = f"Backlog (failed to move to {sprint_name.strip()})"

            # Force description update if needed
            if description_text and description_text.strip():
                time.sleep(3)  # Give Jira time to process
                try:
                    self.update_description(issue_key, description_text)
                    logger.info(f"Description force-updated for {issue_key}")
                except Exception as e:
                    logger.warning(f"Description update failed: {e}")

            # Get final issue details
            final = self.get_issue(
                issue_key, "summary,description,priority,assignee,reporter,status"
            )

            # Get assignee name
            assignee_display_name = (final["fields"]["assignee"] or {}).get(
                "displayName", "Unassigned"
            )

            url_dict = {
                os.getenv("JIRA_BASE_URL"): os.getenv("JIRA_DOMAIN_URL"),
                os.getenv("JIRA_ARK_BASE_URL"): os.getenv("JIRA_AKR_DOMAIN_URL"),
            }
            get_ticket_url = url_dict.get(self.base_url)
            # Build the ticket URL
            ticket_url = f"{get_ticket_url.rstrip('/')}/browse/{issue_key}"

            # 🆕 Format the strict response
            formatted_response = self.format_ticket_creation_response(
                issue_key=issue_key,
                assignee_name=assignee_display_name,
                epic_key=epic_key,  # Will be None if not provided
                jira_url=ticket_url,
                project_key=project_key,
            )

            result = {
                "success": True,
                "message": formatted_response,
                "key": issue_key,
                "summary": final["fields"]["summary"],
                "description": final["fields"].get(
                    "description", "No description provided"
                ),
                "priority": (final["fields"]["priority"] or {}).get("name", "Medium"),
                "assignee": (final["fields"]["assignee"] or {}).get(
                    "displayName", "Unassigned"
                ),
                "reporter": (final["fields"]["reporter"] or {}).get(
                    "displayName", "Unknown"
                ),
                "status": (final["fields"]["status"] or {}).get("name", "To Do"),
                "url": f"{ticket_url}",
                "board_info": board_info,
                "issue_type": normalized_issue_type,
                "sprint": sprint_status,
                "project": project_key,
            }

            # Add story points info if set
            if story_points:
                result["story_points"] = story_points

            # Add epic info if set
            if epic_key:
                result["epic_key"] = epic_key

            # Add detailed reporter information
            if reporter_info["set"]:
                result["reporter_matched"] = True
                result["reporter_match_type"] = reporter_info["match_type"]
                result["reporter_source"] = (
                    f"Slack user '{slack_username}'" if slack_username else "Email"
                )
                if reporter_info.get("confidence"):
                    result["reporter_match_confidence"] = reporter_info["confidence"]
                logger.info(
                    f"✅ Reporter successfully set via {reporter_info['match_type']} match"
                )
            elif reporter_info["error"]:
                result["reporter_match_failed"] = True
                result["reporter_match_error"] = reporter_info["error"]
                logger.warning(f"⚠️ Reporter matching failed: {reporter_info['error']}")

            # Add assignment information if there were issues
            if assignment_info["suggestions"]:
                result["assignment_failed"] = True
                result["user_suggestions"] = assignment_info["suggestions"]
                result["assignment_message"] = (
                    f"Ticket created but could not assign to '{assignee_email}'"
                )

            logger.info(
                f"🎉 Issue {issue_key} created in project {project_key} | "
                f"Status: {result.get('status')} | Sprint: {sprint_status} | "
                f"Reporter: {reporter_info['name']} | Assignee: {assignment_info['assignee_name']}"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Error creating issue: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to create Jira issue: {str(e)}",
            }

    def get_account_id(self, query: str) -> str:
        """Get Jira user account ID."""
        r = self.session.get(
            f"{self.base_url}/rest/api/3/user/search",
            params={"query": query},
            timeout=20,
        )
        r.raise_for_status()
        users = r.json()
        if not users:
            raise RuntimeError(f"No Jira user found for '{query}'.")
        return users[0]["accountId"]

    @staticmethod
    def extract_issue_key_from_response(response_data: str) -> str:
        """Extract issue key from Jira response data."""
        try:
            if not response_data:
                return ""

            # Look for patterns like AI-123, PROJ-456, etc.
            import re

            pattern = r"\b([A-Z]+[-_]\d+)\b"
            matches = re.findall(pattern, response_data)

            if matches:
                # Return the first match (most likely the created/updated issue)
                return matches[0]

            return ""
        except Exception as e:
            logger.error(f"Error extracting issue key: {e}")
            return ""

    def update_issue(
        self,
        issue_key: str,
        summary: str = None,
        description_text: str = None,
        assignee_email: str = None,
        priority_name: str = None,
        due_date: str = None,
        start_date: str = None,
        issue_type_name: str = None,
        labels: list = None,
        sprint_name: str = None,
        status_name: str = None,
        story_points: int = None,  # NEW
        epic_key: str = None,  # NEW
    ) -> dict:
        """Update existing Jira issue with new field values including sprint movement."""
        try:
            # Get current issue to validate it exists
            current_issue = self.get_issue(issue_key, "project,issuetype")
            project_key = current_issue["fields"]["project"]["key"]
            current_issue_type = current_issue["fields"]["issuetype"]["name"]

            # Determine issue type to use
            if issue_type_name:
                normalized_issue_type = self.normalize_issue_type(
                    project_key, issue_type_name
                )
            else:
                normalized_issue_type = current_issue_type

            # Get allowed fields for updates
            allowed = self.get_create_fields(project_key, normalized_issue_type)

            fields = {}

            # Track assignment information
            assignment_info = {
                "assigned": False,
                "assignee_name": "Unassigned",
                "suggestions": None,
            }

            # Update summary
            if summary and "summary" in allowed:
                fields["summary"] = summary.strip().strip("'\"")

            # Update description
            if description_text and "description" in allowed:
                fields["description"] = self.text_to_adf(description_text)

            # Enhanced assignee handling
            if assignee_email is not None and "assignee" in allowed:
                if assignee_email and assignee_email.strip():
                    assignment_result = self.smart_assign_user(
                        project_key, assignee_email.strip()
                    )

                    if assignment_result["success"]:
                        if assignment_result["accountId"]:
                            fields["assignee"] = {"id": assignment_result["accountId"]}
                            assignment_info["assigned"] = True
                            assignment_info["assignee_name"] = assignment_result[
                                "displayName"
                            ]
                            logger.info(
                                f"Successfully assigned {issue_key} to: {assignment_result['displayName']}"
                            )
                        else:
                            fields["assignee"] = None
                            assignment_info["assignee_name"] = "Unassigned"
                            logger.info(f"Leaving {issue_key} unassigned")
                    else:
                        logger.warning(
                            f"Could not assign '{assignee_email}' to {issue_key}: {assignment_result['message']}"
                        )
                        assignment_info["suggestions"] = assignment_result.get(
                            "suggestions", ""
                        )
                else:
                    fields["assignee"] = None
                    assignment_info["assignee_name"] = "Unassigned"
                    logger.info(f"Unassigning {issue_key}")

            # Update priority
            if priority_name and "priority" in allowed:
                try:
                    pr_id = self.get_priority_id_by_name(priority_name)
                    fields["priority"] = {"id": pr_id}
                except Exception as e:
                    logger.warning(f"Could not set priority '{priority_name}': {e}")

            # Update due date
            if due_date and "duedate" in allowed:
                fields["duedate"] = due_date  # Expected format: YYYY-MM-DD
                logger.info(f"Setting due date to: {due_date}")

            # Update issue type
            if (
                issue_type_name
                and normalized_issue_type != current_issue_type
                and "issuetype" in allowed
            ):
                fields["issuetype"] = {"name": normalized_issue_type}

            # Update labels
            if labels is not None and "labels" in allowed:
                fields["labels"] = labels

            # NEW: Update story points
            if story_points is not None:
                try:
                    story_points_field = self.get_story_points_field_id(project_key)
                    if story_points_field in allowed:
                        fields[story_points_field] = story_points
                        logger.info(f"Setting story points to: {story_points}")
                except Exception as e:
                    logger.warning(f"Could not set story points '{story_points}': {e}")

            # NEW: Update epic link
            if epic_key is not None:
                try:
                    epic_link_field = self.get_epic_link_field_id(project_key)
                    if epic_link_field in allowed:
                        fields[epic_link_field] = epic_key if epic_key else None
                        logger.info(
                            f"{'Linking to epic' if epic_key else 'Removing epic link'}: {epic_key}"
                        )
                except Exception as e:
                    logger.warning(f"Could not update epic link '{epic_key}': {e}")

            if not fields and not sprint_name and not status_name:
                return {
                    "success": False,
                    "message": "No valid fields provided for update or fields not allowed for this issue type",
                }

            # Handle regular field updates first
            if fields:
                logger.info(
                    f"Updating issue {issue_key} with fields: {list(fields.keys())}"
                )
                update_url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
                resp = self.session.put(update_url, json={"fields": fields}, timeout=30)

                if not resp.ok:
                    logger.error(f"Update failed: {resp.status_code} - {resp.text}")
                    raise RuntimeError(
                        f"Jira update failed {resp.status_code}: {resp.text}"
                    )

            # Enhanced sprint handling
            sprint_status = None
            sprint_updated = False

            if sprint_name is not None:
                try:
                    sprint_name_clean = sprint_name.lower().strip()

                    if sprint_name_clean in [
                        "backlog",
                        "main backlog",
                        "project backlog",
                    ]:
                        # Move to backlog - clear the sprint field
                        logger.info(f"Moving {issue_key} to backlog")

                        # Try to find and clear the sprint custom field
                        current_issue_full = self.get_issue(issue_key, "*all")
                        sprint_field_id = None

                        # Find the sprint custom field
                        for field_id, field_value in current_issue_full[
                            "fields"
                        ].items():
                            if (
                                field_id.startswith("customfield_")
                                and field_value is not None
                            ):
                                if (
                                    isinstance(field_value, list)
                                    and len(field_value) > 0
                                ):
                                    sprint_item = (
                                        field_value[0] if field_value else None
                                    )
                                    if sprint_item and isinstance(
                                        sprint_item, (str, dict)
                                    ):
                                        if (
                                            "Sprint" in str(sprint_item)
                                            or "sprint" in str(sprint_item).lower()
                                        ):
                                            sprint_field_id = field_id
                                            logger.info(
                                                f"Found sprint field: {field_id}"
                                            )
                                            break

                        if sprint_field_id:
                            # Clear the sprint field to move to backlog
                            clear_sprint_payload = {"fields": {sprint_field_id: None}}
                            response = self.session.put(
                                f"{self.base_url}/rest/api/3/issue/{issue_key}",
                                json=clear_sprint_payload,
                                timeout=30,
                            )

                            if response.ok:
                                sprint_status = "Backlog"
                                sprint_updated = True
                                logger.info(
                                    f"Successfully moved {issue_key} to backlog"
                                )
                            else:
                                logger.warning(
                                    f"Failed to clear sprint field: {response.status_code} - {response.text}"
                                )
                        else:
                            # Try common sprint field IDs
                            common_sprint_fields = [
                                "customfield_10020",
                                "customfield_10014",
                                "customfield_10010",
                            ]
                            for field_id in common_sprint_fields:
                                try:
                                    clear_payload = {"fields": {field_id: None}}
                                    response = self.session.put(
                                        f"{self.base_url}/rest/api/3/issue/{issue_key}",
                                        json=clear_payload,
                                        timeout=30,
                                    )
                                    if response.ok:
                                        sprint_status = "Backlog"
                                        sprint_updated = True
                                        logger.info(
                                            f"Successfully moved {issue_key} to backlog using field {field_id}"
                                        )
                                        break
                                except Exception:
                                    continue

                    else:
                        # Move to specific sprint
                        logger.info(
                            f"Moving {issue_key} to sprint: {sprint_name.strip()}"
                        )

                        board_id = self._get_board_id_for_project(project_key)
                        if not board_id:
                            raise RuntimeError(
                                f"No board found for project {project_key}"
                            )

                        sprint_id = self._get_sprint_id_by_name(
                            board_id, sprint_name.strip()
                        )
                        if not sprint_id:
                            raise RuntimeError(
                                f"{sprint_name.strip()} not found on this board"
                            )

                        self._add_issue_to_sprint(sprint_id, issue_key)
                        sprint_status = sprint_name.strip()
                        sprint_updated = True
                        logger.info(
                            f"Issue {issue_key} moved to sprint {sprint_name.strip()} (id={sprint_id})"
                        )

                except Exception as e:
                    logger.warning(
                        f"Failed to move {issue_key} to sprint {sprint_name}: {e}"
                    )
                    sprint_status = f"Sprint move failed: {str(e)}"

            # Handle status updates
            status_updated = False
            if status_name:
                try:
                    # Get available transitions
                    transitions_url = (
                        f"{self.base_url}/rest/api/3/issue/{issue_key}/transitions"
                    )
                    trans_resp = self.session.get(transitions_url, timeout=20)
                    trans_resp.raise_for_status()
                    transitions = trans_resp.json().get("transitions", [])

                    # Find matching transition
                    target_transition = None
                    for transition in transitions:
                        if transition["to"]["name"].lower() == status_name.lower():
                            target_transition = transition
                            break

                    if target_transition:
                        # Execute the transition
                        transition_data = {
                            "transition": {"id": target_transition["id"]}
                        }
                        trans_url = (
                            f"{self.base_url}/rest/api/3/issue/{issue_key}/transitions"
                        )
                        trans_resp = self.session.post(
                            trans_url, json=transition_data, timeout=20
                        )

                        if trans_resp.ok:
                            status_updated = True
                            logger.info(
                                f"Successfully transitioned {issue_key} to {status_name}"
                            )
                        else:
                            logger.warning(
                                f"Failed to transition issue: {trans_resp.text}"
                            )
                    else:
                        logger.warning(
                            f"No transition found to status '{status_name}' for {issue_key}"
                        )

                except Exception as e:
                    logger.warning(f"Error updating status for {issue_key}: {e}")

            # Get updated issue details
            field_list = "summary,description,priority,assignee,reporter,status,duedate"
            updated = self.get_issue(issue_key, field_list)

            # Determine which fields were actually updated
            updated_fields = []
            if summary:
                updated_fields.append("Summary")
            if description_text:
                updated_fields.append("Description")
            if assignee_email is not None:
                updated_fields.append("Assignee")
            if priority_name:
                updated_fields.append("Priority")
            if due_date:
                updated_fields.append("Due Date")
            if issue_type_name:
                updated_fields.append("Issue Type")
            if labels is not None:
                updated_fields.append("Labels")
            if sprint_updated:
                updated_fields.append("Sprint")
            if status_updated:
                updated_fields.append("Status")
            if story_points is not None:
                updated_fields.append("Story Points")
            if epic_key is not None:
                updated_fields.append("Epic Link")

            # Extract dates for response
            response_due_date = updated["fields"].get("duedate")
            url_dict = {
                os.getenv("JIRA_BASE_URL"): os.getenv("JIRA_DOMAIN_URL"),
                os.getenv("JIRA_ARK_BASE_URL"): os.getenv("JIRA_AKR_DOMAIN_URL"),
            }
            get_ticket_url = url_dict.get(self.base_url)
            # Build the ticket URL
            ticket_url = f"{get_ticket_url.rstrip('/')}/browse/{issue_key}"


            result = {
                "success": True,
                "message": f"Successfully updated Jira issue {issue_key}",
                "key": issue_key,
                "summary": updated["fields"]["summary"],
                "priority": (updated["fields"]["priority"] or {}).get("name", "Medium"),
                "assignee": (updated["fields"]["assignee"] or {}).get(
                    "displayName", "Unassigned"
                ),
                "status": (updated["fields"]["status"] or {}).get("name", "To Do"),
                "url": f"{ticket_url}",
                "updated_fields": updated_fields,
                "due_date": response_due_date,
            }

            # Add sprint info if sprint was updated
            if sprint_updated and sprint_status:
                result["sprint"] = sprint_status

            # Add story points and epic if they were updated
            if story_points is not None:
                result["story_points"] = story_points
            if epic_key is not None:
                result["epic_key"] = epic_key if epic_key else "Removed"

            # Add assignment information if there were issues
            if assignment_info["suggestions"]:
                result["assignment_failed"] = True
                result["user_suggestions"] = assignment_info["suggestions"]
                result["assignment_message"] = (
                    f"Issue updated but could not assign to '{assignee_email}'"
                )

            logger.info(f"Issue {issue_key} updated successfully")
            return result

        except Exception as e:
            logger.error(f"Error updating issue {issue_key}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to update Jira issue {issue_key}: {str(e)}",
            }

    def delete_issue(self, issue_key: str) -> dict:
        """Delete existing Jira issue."""
        try:
            # First, verify the issue exists
            try:
                issue = self.get_issue(issue_key, "summary")
                issue_summary = issue["fields"]["summary"]
            except Exception:
                return {
                    "success": False,
                    "message": f"Issue {issue_key} not found or you don't have permission to view it",
                }

            # Delete the issue
            delete_url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
            resp = self.session.delete(delete_url, timeout=30)

            if not resp.ok:
                logger.error(f"Delete failed: {resp.status_code} - {resp.text}")
                raise RuntimeError(
                    f"Jira delete failed {resp.status_code}: {resp.text}"
                )

            result = {
                "success": True,
                "message": f"Successfully deleted Jira issue {issue_key}",
                "key": issue_key,
                "summary": issue_summary,
            }

            logger.info(f"Issue {issue_key} deleted successfully")
            return result

        except Exception as e:
            logger.error(f"Error deleting issue {issue_key}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to delete Jira issue {issue_key}: {str(e)}",
            }

    def get_channel_name(self, channel_id: str) -> str:
        """Get channel name from Slack API using channel ID."""
        try:
            if not channel_id:
                return "unknown"

            response = client.conversations_info(channel=channel_id)
            if response["ok"]:
                channel = response["channel"]
                # Handle different channel types
                if channel.get("is_im"):
                    return "direct_message"
                elif channel.get("is_mpim"):
                    return "group_message"
                else:
                    return channel.get("name", "unknown")
            else:
                logger.warning(f"Failed to get channel info: {response.get('error')}")
                return "unknown"
        except Exception as e:
            logger.error(f"Error getting channel name for {channel_id}: {e}")
            return "unknown"

    def save_slack_tracking_data(
        self, message_id: str, channel_id: str, channel_name: str, issue_key: str
    ) -> None:
        """Save Slack tracking data to JSON file with duplicate issue key prevention."""
        try:
            file_path = "slack_message.json"

            # Create new record
            new_record = {
                "message_id": message_id,
                "channel_id": channel_id,
                "channel_name": channel_name,
                "issue_key": issue_key,
                "timestamp": datetime.now().isoformat(),
            }

            # Read existing data
            existing_data = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_data = []

            # Check if issue_key already exists
            existing_issue_keys = {item.get("issue_key") for item in existing_data}

            if issue_key in existing_issue_keys:
                logger.info(
                    f"Issue key {issue_key} already exists in tracking data - skipping duplicate"
                )
                return

            # Append new record only if issue_key doesn't exist
            existing_data.append(new_record)

            # Write back to file
            with open(file_path, "w") as f:
                json.dump(existing_data, f, indent=2)

            logger.info(
                f"Saved tracking data for NEW issue {issue_key} in channel {channel_name}"
            )

        except Exception as e:
            logger.error(f"Error saving slack tracking data: {e}")

    def postStatusMsgToSlack(issueKey: str, status_name: str):
        print("Entered in postStatusMsgToSlack")
        emoji_dict = {
            "done": "✅",
            "progress": "🔄",
            "to do": "📝",
            "in progress": "🔄",
        }

        with open("slack_message.json", "r") as file:
            data = json.load(file)

        for item in data:
            if item.get("issue_key") == issueKey:
                print("Issue key matched")
                completed_message = f"The ticket {issueKey} has status: {status_name}"
                channel_id = item.get("channel_id")
                thread_ts = item.get("message_id")

                if not Utils.checkLastMsg(channel_id, thread_ts, completed_message):
                    print("Posting status message to Slack")
                    m_emoji = emoji_dict.get(status_name.lower(), "👋")
                    response = client.chat_postMessage(
                        channel=channel_id,
                        text=f"{m_emoji} {completed_message}",
                        thread_ts=thread_ts,
                    )
                    print(response)
        return True

    def checkLastMsg(channel_id: str, thread_ts: str, complete_msg: str) -> bool:
        response = client.conversations_replies(channel=channel_id, ts=thread_ts)

        last_message = (
            response["messages"][-1]["text"].strip() if response.get("messages") else ""
        )
        return last_message in complete_msg

    def find_reporter_by_slack_username(
        self, project_key: str, slack_username: str
    ) -> dict:
        """
        Find best matching Jira user based on Slack username.
        Uses fuzzy matching to handle variations like 'M Waqas' -> 'Muhammad Waqas'
        """
        try:
            if not slack_username or not slack_username.strip():
                return {
                    "success": False,
                    "accountId": None,
                    "message": "No slack username provided",
                }

            users = self.get_project_users(project_key)
            if not users:
                return {
                    "success": False,
                    "accountId": None,
                    "message": "No users found in project",
                }

            slack_username_lower = slack_username.lower().strip()
            slack_parts = set(slack_username_lower.split())

            best_match = None
            best_score = 0

            for user in users:
                display_name = user.get("displayName", "").lower()
                email = user.get("emailAddress", "").lower()

                # Exact match (highest priority)
                if (
                    slack_username_lower == display_name
                    or slack_username_lower == email.split("@")[0]
                ):
                    return {
                        "success": True,
                        "accountId": user["accountId"],
                        "displayName": user["displayName"],
                        "matchType": "exact",
                        "message": f"Exact match found: {user['displayName']}",
                    }

                # Word-based matching
                display_parts = set(display_name.split())
                email_parts = set(email.split("@")[0].split("."))

                # Calculate match score
                score = 0

                # Check if all slack name parts are in display name
                matches_in_display = sum(
                    1 for part in slack_parts if part in display_name
                )
                matches_in_email = sum(1 for part in slack_parts if part in email)

                # Word overlap scoring
                word_overlap = len(slack_parts & display_parts)
                email_overlap = len(slack_parts & email_parts)

                score = max(
                    matches_in_display * 2,  # Substring matches worth more
                    matches_in_email * 2,
                    word_overlap * 3,  # Word matches worth most
                    email_overlap * 3,
                )

                # Bonus for partial matches (like "fahad" in "fahad ahmed")
                if (
                    slack_username_lower in display_name
                    or slack_username_lower in email
                ):
                    score += 5

                # Update best match
                if score > best_score:
                    best_score = score
                    best_match = user

            if best_match and best_score >= 2:  # Minimum threshold
                return {
                    "success": True,
                    "accountId": best_match["accountId"],
                    "displayName": best_match["displayName"],
                    "matchType": "fuzzy",
                    "score": best_score,
                    "message": f"Matched Slack user '{slack_username}' to Jira user '{best_match['displayName']}' (confidence: {best_score})",
                }

            logger.warning(
                f"No good match found for Slack username '{slack_username}' in project {project_key}"
            )
            return {
                "success": False,
                "accountId": None,
                "message": f"Could not find a good match for Slack user '{slack_username}' in Jira",
            }

        except Exception as e:
            logger.error(f"Error matching Slack username '{slack_username}': {e}")
            return {
                "success": False,
                "accountId": None,
                "message": f"Error finding reporter: {str(e)}",
            }

    def build_slack_thread_link(self, channel_id: str, message_ts: str) -> str:
        """
        Build a Slack thread permalink from channel ID and message timestamp.

        Args:
            channel_id: Slack channel ID (e.g., C12345678)
            message_ts: Message timestamp (e.g., 1234567890.123456)

        Returns:
            str: Slack permalink or empty string if failed
        """
        try:
            if not channel_id or not message_ts:
                logger.warning("Missing channel_id or message_ts for Slack link")
                return ""

            # Get workspace info to build the permalink
            response = client.conversations_info(channel=channel_id)

            if not response.get("ok"):
                logger.warning(f"Failed to get channel info: {response.get('error')}")
                return ""

            # Get team/workspace info
            team_response = client.team_info()
            if not team_response.get("ok"):
                logger.warning("Failed to get team info")
                return ""

            team_domain = team_response["team"]["domain"]

            # Format: https://workspace.slack.com/archives/CHANNEL_ID/pMESSAGE_TS
            # Convert message_ts format: 1234567890.123456 -> p1234567890123456
            message_id = message_ts.replace(".", "")
            slack_link = (
                f"https://{team_domain}.slack.com/archives/{channel_id}/p{message_id}"
            )

            logger.info(f"Generated Slack link: {slack_link}")
            return slack_link

        except Exception as e:
            logger.error(f"Error building Slack thread link: {e}")
            return ""

    def append_slack_link_to_description(
        self, description_text: str, slack_link: str
    ) -> str:
        """
        Append Slack thread link to the end of description text.

        Args:
            description_text: Original description text
            slack_link: Slack thread permalink

        Returns:
            str: Description with Slack link appended
        """
        if not slack_link:
            return description_text

        # Add separator and link at the end
        separator = "\n\n---\n"
        slack_section = f"📎 Source: {slack_link}"

        return f"{description_text}{separator}{slack_section}"

    def format_ticket_creation_response(
        self,
        issue_key: str,
        assignee_name: str,
        epic_key: str = None,
        jira_url: str = None,
        project_key: str = str(os.getenv("Default_Project")),
    ) -> str:
        """
        Format a strict, consistent ticket creation response with Slack hyperlink formatting.

        Args:
            issue_key: Created ticket key (e.g., "AI-3423")
            assignee_name: Name of assigned user
            epic_key: Epic key if linked, None otherwise
            jira_url: Full Jira URL to the ticket
            project_key: Project key for epic suggestion

        Returns:
            str: Formatted response string with Slack hyperlink
        """
        # Build ticket URL if not provided
        if not jira_url:
            jira_url = f"{self.base_url.rstrip('/')}/browse/{issue_key}"

        # Format epic status
        if epic_key:
            epic_status = epic_key
        else:
            epic_status = "unknown - would you like a list of epics to choose from?"

        # Build the strict response format with Slack hyperlink and spacing
        # Slack hyperlink format: <URL|Display Text>
        response = (
            f"Ticket created: <{jira_url}|{issue_key}>\n"
            f"\n"  # Add blank line for spacing
            f"Assigned to: {assignee_name}\n"
            f"\n"  # Add blank line for spacing
            f"Epic: {epic_status}"
        )

        return response

    async def _get_embedding(self, text: str) -> list:
        """Get embedding using OpenAI API"""
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await asyncio.to_thread(
            openai_client.embeddings.create,
            input=text,
            model="text-embedding-3-small",
            dimensions=1024,
        )
        return response.data[0].embedding

    async def search_confluence_knowledge(self, user_query: str):
        """Search knowledge base and return retrieved content"""
        try:
            if not user_query.strip():
                return {"status": "error", "message": "Query cannot be empty"}
            qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=False,
                check_compatibility=False,
            )
            QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

            query_vector = await self._get_embedding(user_query)
            search_results = await asyncio.to_thread(
                qdrant_client.search,
                collection_name=QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=8,
                with_payload=True,
            )
            if not search_results:
                return {
                    "status": "success",
                    "retrieved_content": [],
                    "total_pages_searched": 0,
                }

            retrieved_content = []

            for i, result in enumerate(search_results, 1):
                payload = result.payload
                content = payload.get("text", "")
                if content.strip():  # Only add non-empty content
                    retrieved_content.append(
                        {
                            "doc_id": i,
                            "title": payload.get("title", "Unknown"),
                            "space": payload.get(
                                "space", payload.get("project", "Unknown")
                            ),
                            "content": content,
                            "url": payload.get("url", ""),
                            "score": result.score,
                        }
                    )

            return {
                "status": "success",
                "query": user_query,
                "retrieved_content": retrieved_content,
                "total_pages_searched": len(retrieved_content),
            }

        except Exception as e:
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}
