import os, re
import time
import json
import logging
from datetime import datetime, timedelta, timezone
from slack_sdk import WebClient
from dotenv import load_dotenv
from slack_sdk.errors import SlackApiError

load_dotenv()
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
logger = logging.getLogger(__name__)


class Utils:
    """Pure utility class - all intelligence handled by LangGraph agent."""

    def __init__(self, base_url, email, token, session):
        self.base_url = base_url
        self.email = email
        self.token = token
        self.session = session

    # ========================================
    # SLACK INTEGRATION (RAW DATA ONLY)
    # ========================================

    def extract_chat(self, channel_id):
        """Extract channel messages + all thread replies, sorted like Slack."""
        if not channel_id:
            return ""

        # ----- time window: today -----
        today = datetime.now().astimezone()
        day_start = today.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        oldest = day_start.timestamp()
        latest = min(day_end.timestamp(), today.timestamp())

        # ----- fetch channel history (paginate) -----
        messages = []
        cursor = None
        try:
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
                    # skip deleted/system noise
                    if m.get("subtype") == "message_deleted":
                        continue

                    # replies that were surfaced in-channel are skipped; we fetch under parent
                    if m.get("thread_ts") and m["thread_ts"] != m["ts"]:
                        continue

                    # parent/top-level
                    m["_depth"] = 0
                    m["_parent_ts"] = m["ts"]
                    messages.append(m)

                    # ----- fetch thread replies (paginate) -----
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
                                # gentle retry on rate limit
                                if e.response.status_code == 429:
                                    time.sleep(
                                        int(e.response.headers.get("Retry-After", "2"))
                                    )
                                    continue
                                break

                            r_msgs = r.get("messages", [])
                            for rep in r_msgs:
                                # replies payload includes the parent; skip it
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

            # ----- order like Slack: parent → replies (chronological) -----
            messages.sort(
                key=lambda msg: (
                    float(msg.get("_parent_ts") or msg["ts"]),
                    msg["_depth"],
                    float(msg["ts"]),
                )
            )

            # save debug dump as JSON (not str(...))

            if not messages:
                return ""

            # ----- helpers for formatting -----
            user_cache = {}

            def get_user_name(user_id, bot_profile=None):
                # prefer bot profile name if present
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
                # <@U123> -> @Name
                t = re.sub(
                    r"<@([A-Z0-9]+)>", lambda m: f"@{get_user_name(m.group(1))}", t
                )
                # <#C123|channel-name> -> #channel-name
                t = re.sub(r"<#([A-Z0-9]+)\|([^>]+)>", r"#\2", t)
                # <https://url|label> -> https://url
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

            # ----- format transcript (Slack-like; replies indented) -----
            formatted_lines = []
            for msg in messages:
                txt = clean_text(extract_text(msg))
                if not txt:
                    continue
                ts = datetime.fromtimestamp(float(msg["ts"]), tz=timezone.utc).strftime(
                    "%H:%M"
                )
                uname = get_user_name(msg.get("user"), msg.get("bot_profile"))
                indent = "    " * msg["_depth"]
                # optional: a visual pointer for replies
                arrow = "↳ " if msg["_depth"] else ""
                formatted_lines.append(f"{indent}{ts} {uname}: {arrow}{txt}")

            # keep last 200 lines for brevity; adjust as you like
            return "\n".join(formatted_lines[-200:])

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
        """Return project key by name or key."""
        r = self.session.get(f"{self.base_url}/rest/api/3/project/search", timeout=20)
        r.raise_for_status()
        projects = r.json().get("values", [])

        for p in projects:
            if (
                p["key"].lower() == name_or_key.lower()
                or p["name"].lower() == name_or_key.lower()
            ):
                return p["key"]
        raise RuntimeError(f"Project '{name_or_key}' not found.")

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

    def _get_board_id_for_project(self, project_key: str) -> int:
        """Find a board for this project (prefer scrum)."""
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
        """
        Get list of sprints for a project. Use this when you need sprint options.

        Args:
            project_name_or_key: Jira project key (e.g., "SCRUM", "LUNA_TICKETS")

        Returns:
            dict: Sprint list information
        """
        try:
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
        """
        Helper tool to get the project key from an issue key.
        Use this when you need to know which project an issue belongs to.

        Args:
            issue_key: The ticket ID (e.g., "LT-23", "SCRUM-456")

        Returns:
            dict: Project information for the issue
        """
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
    # CRUD OPERATIONS (KEEP YOUR EXISTING IMPLEMENTATIONS)
    # ========================================

    def create_issue_implementation(
        self,
        project_name_or_key: str = "",
        summary: str = "",
        description_text: str = "",
        assignee_email: str = "",
        priority_name: str = None,
        reporter_email: str = None,
        issue_type_name: str = None,
        sprint_name: str = None,
        force_update_description_after_create: bool = True,
    ) -> dict:
        """Create a new Jira ticket/issue with validation."""
        logger.info(
            f"Creating issue with context - assignee: {assignee_email}, summary: {summary}"
        )

        if issue_type_name is None:
            issue_type_name = "Story"

        # Use default project if not provided
        if not project_name_or_key or project_name_or_key.strip() == "":
            project_name_or_key = "AI"
            logger.info(f"No project specified, using default: AI")

        # Handle sprint selection with user fallback
        if not sprint_name:
            sprint_info = self.get_default_sprint_for_project(project_name_or_key)

            if sprint_info["has_default"]:
                sprint_name = sprint_info["sprint_name"]
                logger.info(f"Using default upcoming sprint: {sprint_name}")
            else:
                logger.info(
                    f"No upcoming sprint available for {project_name_or_key}, asking user"
                )
                return {
                    "success": False,
                    "needs_sprint_selection": True,
                    "message": f"Which sprint should I add this ticket to? Here are the available options:\n\n{sprint_info['sprint_list']}\n\nPlease specify the exact sprint name or 'backlog'.",
                    "project": project_name_or_key,
                    "assignee": assignee_email,
                    "sprint_options": sprint_info["sprint_list"],
                }

        # Generate default summary if empty
        if not summary or summary.strip() == "":
            summary = "New ticket created via AI assistant"

        # Generate default description if empty
        if not description_text or description_text.strip() == "":
            description_text = "This ticket was created through the AI assistant and needs further details to be added."

        try:
            project_key = self.resolve_project_key(project_name_or_key)
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

            # Set priority
            if priority_name and "priority" in allowed:
                try:
                    pr_id = self.get_priority_id_by_name(priority_name)
                    fields["priority"] = {"id": pr_id}
                except Exception as e:
                    logger.warning(f"Could not set priority '{priority_name}': {e}")

            # Set reporter
            if reporter_email and "reporter" in allowed:
                try:
                    reporter_id = self.get_account_id(reporter_email)
                    fields["reporter"] = {"id": reporter_id}
                except Exception as e:
                    logger.warning(f"Could not set reporter '{reporter_email}': {e}")

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
            logger.info(f"Successfully created issue: {issue_key}")

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
            final = self.get_issue(issue_key)

            result = {
                "success": True,
                "message": f"Successfully created Jira issue {issue_key}",
                "key": issue_key,
                "summary": final["fields"]["summary"],
                "description": final["fields"].get(
                    "description", "No description provided"
                ),
                "priority": (final["fields"]["priority"] or {}).get("name", "Medium"),
                "assignee": (final["fields"]["assignee"] or {}).get(
                    "displayName", "Unassigned"
                ),
                "status": (final["fields"]["status"] or {}).get("name", "To Do"),
                "url": f"{self.base_url.rstrip('/')}/browse/{issue_key}",
                "board_info": board_info,
                "issue_type": normalized_issue_type,
                "sprint": sprint_status,
                "project": project_key,
            }

            # Add assignment information if there were issues
            if assignment_info["suggestions"]:
                result["assignment_failed"] = True
                result["user_suggestions"] = assignment_info["suggestions"]
                result["assignment_message"] = (
                    f"Ticket created but could not assign to '{assignee_email}'"
                )

            logger.info(
                f"Issue {issue_key} created in project {project_key} with status: {result.get('status')} in sprint: {sprint_status}"
            )
            return result

        except Exception as e:
            logger.error(f"Error creating issue: {e}")
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

            # Extract dates for response
            response_due_date = updated["fields"].get("duedate")

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
                "url": f"{self.base_url.rstrip('/')}/browse/{issue_key}",
                "updated_fields": updated_fields,
                "due_date": response_due_date,
            }

            # Add sprint info if sprint was updated
            if sprint_updated and sprint_status:
                result["sprint"] = sprint_status

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

    def extract_issue_key_from_response(self, response_data: str) -> str:
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
