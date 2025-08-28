import os
import time
import requests
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import logging
from .schemas import UserQuery
from dotenv import load_dotenv
from slack_sdk import WebClient
from datetime import datetime, timedelta, timezone

load_dotenv()
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

logger = logging.getLogger(__name__)


class JiraService:
    """Service class for creating Jira tickets using agentic AI."""

    def __init__(self):
        """Initialize Jira service with configuration and dependencies."""
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.base_url = os.getenv("JIRA_BASE_URL")
        self.email = os.getenv("JIRA_EMAIL")
        self.token = os.getenv("JIRA_TOKEN")
        self.default_issue_type = "Task"
        self.default_project = "AI-Jira"  # Set your default project here

        # Session for Jira API calls
        self.session = requests.Session()
        self.session.auth = (self.email, self.token)
        self.session.headers.update(
            {"Accept": "application/json", "Content-Type": "application/json"}
        )

        self._agent = None

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

    def extractChat(self, channel_id):
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

    def _get_agent(self):
        """Create LangGraph agent for ticket creation."""
        if self._agent is None:
            self._agent = create_react_agent(
                model=self.model,
                tools=[self.create_issue_sync],
                prompt="""
You are a friendly and helpful Jira assistant. Your primary job is to create Jira tickets from actionable work items, but you should also respond naturally to casual conversation.

## BEHAVIOR GUIDELINES:
- Respond naturally and conversationally to greetings, casual chat, and friendly messages
- Only mention "no actionable items" when someone explicitly asks you to create tickets or analyze work items
- For casual greetings like "hello", "hi", "good morning", respond warmly and ask how you can help
- Be helpful and human-like in all interactions

## WHAT TO LOOK FOR (for ticket creation):
Look for ANY of the following patterns in chat conversations:

### 1. DIRECT REQUESTS & TASKS
- "I need help with..."
- "Can someone create/build/fix..."
- "We need to implement..."
- "Please set up..."
- "Could you look into..."

### 2. PROBLEMS & ISSUES
- Bug reports or system problems
- Things that are broken or not working
- Performance issues
- User complaints

### 3. QUESTIONS REQUIRING INVESTIGATION
- "Do we have [system/process] still running?"
- "How should we handle [situation]?"
- "What's the status of [project]?"
- "Need advice on how to do..."

### 4. PROJECT REQUIREMENTS & PLANNING
- Feature requests
- New project requirements
- Process improvements needed
- Documentation requests

### 5. FOLLOW-UP ACTIONS
- Items mentioned that need follow-up
- Decisions that require implementation
- Action items from discussions

## IMPORTANT RULES:
1. **No Duplicates**: Before creating a new issue, scan chat history for Jira bot responses.  
- If the exact or highly similar ticket already exists in chat history with an *Issue Key*, DO NOT create it again.  
- Instead, return: "This issue has already been created: [ISSUE-123](URL)".

2. **Multiple Items**: If multiple actionable items exist, call `create_issue_sync` multiple times until ALL distinct tickets are created.

3. **Priority Assignment**:
    - High: Urgent requests, blocking issues, specific deadlines
    - Medium: General tasks, investigations, improvements  
    - Low: Documentation, nice-to-have features

4. **Type Assignment**:
    - Bug: Something is broken/not working
    - Task: General work items, investigations, setups
    - Story: New features, enhancements, user-facing improvements

## RESPONSE PATTERNS:

### For Casual Greetings:
- "Hello! I'm here to help you with Jira tickets and work items. What can I do for you today?"
- "Hi there! How can I assist you with your tasks or projects?"
- "Good morning! Ready to help you track any work items or create tickets. What's on your mind?"

### For General Questions:
- Answer helpfully and conversationally
- Offer to create tickets if work items are mentioned
- Ask clarifying questions when needed

### For Work Items Found:
Create tickets and respond with the format below.

## OUTPUT FORMAT:
If actionable items found and NOT already in history, call create_issue_sync with:
- project_name_or_key: "AI-Jira" (or detected project name)
- summary: Clear, actionable title without quotes
- description_text: Detailed description based on chat context
- assignee_email: [if mentioned, otherwise blank] 
- priority_name: High/Medium/Low based on urgency
- issue_type_name: Bug/Task/Story based on content
- reporter_email: [if identifiable, otherwise blank]

## RESPONSE FORMAT:
**IMPORTANT: Use Slack markdown formatting with single asterisks (*) for bold text**

The *[issue type]* ticket has been successfully created. Here are the details:
- *Issue Key*: [ISSUE-123](URL)
- *Summary*: [summary text]
- *Description*: [description text]
- *Assignee*: [assignee name or "Unassigned"]
- *Priority*: [priority level]
- *Status*: [current status]

All bold text must use single asterisks (*text*) for proper Slack formatting.

## REMEMBER:
- Be conversational and helpful in all interactions
- Only create tickets when there are clear actionable work items
- Respond naturally to casual conversation without mentioning "no actionable items"
- Ask follow-up questions when helpful
- Be friendly and approachable
""",
            )
        return self._agent

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
        if not issue_type_name:
            return self.default_issue_type

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

        return self.default_issue_type

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

    def create_issue_sync(
        self,
        project_name_or_key: str,
        summary: str,
        description_text: str,
        assignee_email: str,
        priority_name: str = None,
        reporter_email: str = None,
        issue_type_name: str = None,
        force_update_description_after_create: bool = True,
    ) -> dict:
        """Create Jira issue - main function used by the agent."""
        if issue_type_name is None:
            issue_type_name = self.default_issue_type

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

            # Set assignee
            if "assignee" in allowed and assignee_email:
                try:
                    assignee_id = self.get_account_id(assignee_email)
                    fields["assignee"] = {"id": assignee_id}
                except Exception as e:
                    logger.warning(f"Could not set assignee '{assignee_email}': {e}")

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
                fields["labels"] = ["created-by-bot"]

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

            # Force description update if needed
            if (
                force_update_description_after_create
                and description_text
                and "description" not in allowed
            ):
                time.sleep(2)
                try:
                    self.update_description(issue_key, description_text)
                    logger.info(f"Description updated for {issue_key}")
                except Exception as e:
                    logger.warning(f"Description update failed: {e}")

            # Get final issue details
            final = self.get_issue(issue_key)

            # Return comprehensive information
            result = {
                "success": True,
                "message": f"Successfully created Jira issue {issue_key}",
                "key": issue_key,
                "summary": final["fields"]["summary"],
                "priority": (final["fields"]["priority"] or {}).get("name"),
                "assignee": (final["fields"]["assignee"] or {}).get("displayName"),
                "status": (final["fields"]["status"] or {}).get("name"),
                "url": f"{self.base_url}/browse/{issue_key}",
                "board_info": board_info,
                "issue_type": normalized_issue_type,
            }

            logger.info(
                f"Issue {issue_key} created with status: {result.get('status')}"
            )
            return result

        except Exception as e:
            logger.error(f"Error creating issue: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to create Jira issue: {str(e)}",
            }

    async def process_query(
        self, user_query: UserQuery, channel_id: str = None
    ) -> dict:
        """Process Jira query from user input - main entry point."""
        try:
            print(f"Channel ID: {channel_id}")
            logger.info(f"Processing query from channel: {channel_id}")

            agent = self._get_agent()
            chat_history_string = self.extractChat(channel_id)

            # Prepare messages for the agent
            chat_messages = []
            if chat_history_string and chat_history_string.strip():
                chat_messages.append(
                    {
                        "role": "user",
                        "content": f"create a jira ticket based on the below chat_history. \n\n Chat History: {chat_history_string}",
                    }
                )

            # Add the current user query
            chat_messages.append({"role": "user", "content": user_query.query})

            # Process with agent
            result = agent.invoke({"messages": chat_messages})

            if result and "messages" in result and len(result["messages"]) > 0:
                final_message = result["messages"][-1]
                response_content = final_message.content

                # Format the response for Slack markdown
                formatted_response = self.format_for_slack(response_content)

                return {
                    "success": True,
                    "message": "Jira operation completed",
                    "data": formatted_response,  # Use formatted response
                    "query": user_query.query,
                }
            else:
                return {
                    "success": False,
                    "message": "No response generated from Jira processing",
                    "data": None,
                    "query": user_query.query,
                }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to process Jira query: {str(e)}",
                "data": None,
                "query": user_query.query,
            }
