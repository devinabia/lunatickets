import os
import requests
import logging
from datetime import datetime, timedelta, timezone
from slack_sdk import WebClient
from dotenv import load_dotenv

load_dotenv()
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
logger = logging.getLogger(__name__)


class Utils:
    """Utility class for helper functions used by JiraService."""

    def __init__(self, base_url, email, token, session):
        self.base_url = base_url
        self.email = email
        self.token = token
        self.session = session

    def get_project_users(self, project_key: str, max_results: int = 50) -> list:
        """
        Get list of users who can be assigned to issues in a specific project.
        Returns list of user objects with displayName, emailAddress, and accountId.
        """
        try:
            # Get users assignable to issues in the project
            r = self.session.get(
                f"{self.base_url}/rest/api/3/user/assignable/search",
                params={"project": project_key, "maxResults": max_results},
                timeout=20,
            )
            r.raise_for_status()
            users = r.json()

            # Format user data for easier consumption
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
        """
        Find a user in the project by display name or email address.
        Returns user data if found, None otherwise.
        """
        try:
            users = self.get_project_users(project_key)

            query_lower = query.lower().strip()

            # First try exact matches
            for user in users:
                if user["emailAddress"].lower() == query_lower:
                    return user
                if user["displayName"].lower() == query_lower:
                    return user

            # Then try partial matches
            for user in users:
                if query_lower in user["displayName"].lower():
                    return user
                if query_lower in user["emailAddress"].lower():
                    return user

            logger.warning(
                f"No user found for query '{query}' in project {project_key}"
            )
            return None

        except Exception as e:
            logger.error(f"Error finding user '{query}' in project {project_key}: {e}")
            return None

    def get_user_suggestions_text(self, project_key: str, limit: int = 10) -> str:
        """
        Get a formatted text list of available users for assignment suggestions.
        Used when user assignment fails or when listing available users.
        """
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
        """
        Intelligently assign a user based on input. Returns assignment info.

        Args:
            project_key: Jira project key
            assignee_input: User input (name, email, or partial match)

        Returns:
            dict with keys: success, accountId, displayName, message, suggestions
        """
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

            # Try to find the user
            user = self.find_user_by_name_or_email(project_key, assignee_input)

            if user:
                return {
                    "success": True,
                    "accountId": user["accountId"],
                    "displayName": user["displayName"],
                    "message": f"Found user: {user['displayName']}",
                }
            else:
                # User not found, provide suggestions
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

    def format_for_slack(self, text: str) -> str:
        """Format response text for Slack markdown."""
        if not text:
            return text

        import re

        # Convert markdown links [text](url) to Slack format <url|text>
        text = re.sub(r"\[([^\]]+?)\]\(([^)]+?)\)", r"<\2|\1>", text)

        # Convert double asterisks (standard markdown bold) to single asterisks (Slack bold)
        # Use regex to avoid replacing asterisks that are already single
        text = re.sub(r"(?<!\*)\*\*(?!\*)([^*]+?)(?<!\*)\*\*(?!\*)", r"*\1*", text)

        # Ensure proper formatting for common patterns
        patterns = [
            (r"\*\*Issue Key\*\*:", r"*Issue Key*:"),
            (r"\*\*Summary\*\*:", r"*Summary*:"),
            (r"\*\*Description\*\*:", r"*Description*:"),
            (r"\*\*Assignee\*\*:", r"*Assignee*:"),
            (r"\*\*Priority\*\*:", r"*Priority*:"),
            (r"\*\*Status\*\*:", r"*Status*:"),
            (r"\*\*([^*]+?)\*\*", r"*\1*"),  # Any other double asterisk patterns
        ]

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)

        return text

    def extract_chat(self, channel_id):
        """Extract chat history from Slack channel for context."""
        date_str = datetime.now().date().strftime("%Y-%m-%d")
        day_start = datetime.strptime(date_str, "%Y-%m-%d").astimezone()
        day_end = day_start + timedelta(days=1)

        oldest = day_start.timestamp()
        latest = min(day_end.timestamp(), datetime.now().astimezone().timestamp())

        print("Oldest", datetime.utcfromtimestamp(oldest).strftime("%Y-%m-%d %H:%M:%S"))
        print("Latest", datetime.utcfromtimestamp(latest).strftime("%Y-%m-%d %H:%M:%S"))

        # Fetch messages
        resp = client.conversations_history(
            channel=channel_id,
            oldest=str(oldest),
            latest=str(latest),
            inclusive=True,
            limit=200,
        )

        # Format messages
        formatted_messages = []
        for msg in resp["messages"]:
            ts = datetime.fromtimestamp(float(msg["ts"]), tz=timezone.utc)
            time_str = ts.strftime("%H:%M:%S")
            user = msg.get("user", "bot")
            text = msg.get("text", "")
            formatted_messages.append(f"[{time_str}] {user}: {text}")

        # Join all messages with newlines (reversed for chronological order)
        return "\n".join(reversed(formatted_messages))

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

    def get_priority_id_by_name(self, name: str) -> str:
        """Get priority ID by name."""
        r = self.session.get(f"{self.base_url}/rest/api/3/priority", timeout=20)
        r.raise_for_status()
        for pr in r.json():
            if pr["name"].lower() == name.lower():
                return pr["id"]
        raise RuntimeError(f"Priority '{name}' not found.")

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

    def normalize_issue_type(self, project_key: str, issue_type_name: str) -> str:
        """Normalize issue type name to match Jira's expectations."""
        default_issue_type = "Task"

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

        # If no match found, use default or first available
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
            # Return common fields as fallback
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

    def update_description(self, issue_key: str, description_text: str) -> None:
        """Update issue description."""
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        body = {"fields": {"description": self.text_to_adf(description_text)}}
        r = self.session.put(url, json=body, timeout=20)
        if not r.ok:
            raise RuntimeError(f"Update description failed {r.status_code}: {r.text}")

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

    def update_issue(
        self,
        issue_key: str,
        summary: str = None,
        description_text: str = None,
        assignee_email: str = None,
        priority_name: str = None,
        due_date: str = None,
        start_date: str = None,  # Keep parameter but ignore it
        issue_type_name: str = None,
        labels: list = None,
    ) -> dict:
        """Update existing Jira issue with new field values."""
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

            # Track assignment information (same pattern as create_issue_sync)
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

            # ENHANCED ASSIGNEE HANDLING (same pattern as create_issue_sync)
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
                            # If accountId is None, leave unassigned (which is success)
                            fields["assignee"] = None
                            assignment_info["assignee_name"] = "Unassigned"
                            logger.info(f"Leaving {issue_key} unassigned")
                    else:
                        # Assignment failed, but continue with other updates
                        logger.warning(
                            f"Could not assign '{assignee_email}' to {issue_key}: {assignment_result['message']}"
                        )
                        assignment_info["suggestions"] = assignment_result.get(
                            "suggestions", ""
                        )
                else:
                    # Empty assignee_email means unassign
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

            # START DATE HANDLING REMOVED
            # Log if start date was requested but skip processing
            if start_date:
                logger.info(
                    f"Start date requested ({start_date}) but start date handling is disabled"
                )

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

            if not fields:
                return {
                    "success": False,
                    "message": "No valid fields provided for update or fields not allowed for this issue type",
                }

            logger.info(
                f"Updating issue {issue_key} with fields: {list(fields.keys())}"
            )

            # Update the issue
            update_url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
            resp = self.session.put(update_url, json={"fields": fields}, timeout=30)

            if not resp.ok:
                logger.error(f"Update failed: {resp.status_code} - {resp.text}")
                raise RuntimeError(
                    f"Jira update failed {resp.status_code}: {resp.text}"
                )

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
            # Remove start date from updated fields since we're not handling it
            if issue_type_name:
                updated_fields.append("Issue Type")
            if labels is not None:
                updated_fields.append("Labels")

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
                # Remove start_date from response since we're not handling it
            }

            # Add assignment information if there were issues (same as create_issue_sync)
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
