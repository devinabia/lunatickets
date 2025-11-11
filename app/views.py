import os
import requests
import logging
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from .schemas import UserQuery
from .utilities.utils import Utils
from dotenv import load_dotenv
from fastapi.responses import PlainTextResponse
from openai import OpenAI
import asyncio
from qdrant_client import QdrantClient
from typing import List, Dict
from app.utilities.utils import JIRA_ACCOUNTS

load_dotenv()
logger = logging.getLogger(__name__)
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


class JiraService:
    """Simplified Jira service with single React agent for all CRUD operations."""

    def __init__(self):
        """Initialize Jira service with single agent and all tools."""
        self.model = ChatOpenAI(
            model="gpt-4o-2024-08-06",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # ❌ REMOVE THESE - No longer needed, Utils has them
        # self.base_url = os.getenv("JIRA_BASE_URL")
        # self.email = os.getenv("JIRA_EMAIL")
        # self.token = os.getenv("JIRA_TOKEN")

        self.default_issue_type = "Task"
        self.default_project = os.getenv("Default_Project")

        # ✅ Create session and utils ONLY - single source of truth
        session = requests.Session()
        default_config = Utils.get_account_config("default")
        session.auth = (default_config["email"], default_config["token"])
        session.headers.update(
            {"Accept": "application/json", "Content-Type": "application/json"}
        )

        # Initialize utilities with default account
        self.utils = Utils(
            default_config["base_url"],
            default_config["email"],
            default_config["token"],
            session,
        )

        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=False,
            check_compatibility=False,
        )

        # Create single agent with all CRUD tools + Confluence search
        self.jira_agent = create_react_agent(
            model=self.model,
            tools=[
                self.detect_jira_account_sync,
                self.create_issue_sync,
                self.update_issue_sync,
                self.delete_issue_sync,
                self.get_sprint_list_sync,
                self.get_project_from_issue_sync,
                self.get_project_assignable_users_sync,
                self.get_project_epics_sync,
                self.search_confluence_knowledge_sync,
            ],
            prompt=self._get_unified_prompt(),
        )

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

    def detect_jira_account_sync(self, user_query: str) -> dict:
        """
        AI tool to detect which Jira account to use based on user query.
        If no account mentioned, maintains current account (sticky behavior).
        Uses word-boundary matching to avoid false positives.
        """
        try:
            import re

            query_lower = user_query.lower()
            current_account = self.utils.current_account
            logger.info(f"Current account before detection: {current_account}")

            # Check each account for mentions (using word boundaries)
            for account_key, config in JIRA_ACCOUNTS.items():
                account_name = config["name"].lower()
                project_key = config.get("project_key", "").lower()

                # Use word boundary regex to avoid false matches
                # \b ensures we match whole words only
                account_pattern = r"\b" + re.escape(account_key) + r"\b"
                name_pattern = r"\b" + re.escape(account_name) + r"\b"
                project_pattern = r"\b" + re.escape(project_key) + r"\b"

                if (
                    re.search(account_pattern, query_lower)
                    or re.search(name_pattern, query_lower)
                    or re.search(project_pattern, query_lower)
                ):
                    logger.info(f"🎯 Detected Jira account: {account_key}")

                    # Switch to detected account
                    switch_success = self.utils.switch_account(account_key)

                    if not switch_success:
                        logger.warning(f"Failed to switch to account: {account_key}")
                        return {
                            "success": False,
                            "account": current_account,
                            "error": f"Could not switch to {account_key}, staying on {current_account}",
                        }

                    return {
                        "success": True,
                        "account": account_key,
                        "account_name": config["name"],
                        "project_key": config["project_key"],
                        "message": f"✅ Switched to {config['name']} account",
                    }

            # No account mentioned - STAY on current account
            logger.info(
                f"📌 No account mentioned in query, staying on current account: {current_account}"
            )

            current_config = JIRA_ACCOUNTS[current_account]
            return {
                "success": True,
                "account": current_account,
                "account_name": current_config["name"],
                "project_key": current_config["project_key"],
                "message": f"Continuing with {current_config['name']} account",
            }

        except Exception as e:
            logger.error(f"Error detecting account: {e}")
            return {
                "success": False,
                "account": self.utils.current_account,
                "error": str(e),
            }

    def _get_unified_prompt(self) -> str:
        """Enhanced unified prompt with strict validation requirements."""

        accounts = ", ".join(
            [f"{v['name']} ({v['project_key']})" for k, v in JIRA_ACCOUNTS.items()]
        )

        return f"""
        You are a Jira assistant that helps create, update, and manage tickets through natural conversation.

        🏢 ACCOUNTS: {accounts}

        🔴 CRITICAL RULES

        0. ACCOUNT DETECTION (ALWAYS FIRST):
        - Call detect_jira_account_sync(user_query) before any Jira operation
        - Account stays active until explicitly changed

        1. THREAD CONTEXT & TICKET MEMORY:
        - "--- 📜 Previous Chat ---" = Historical (reference only, ignore these tickets)
        - "--- 💬 Current Thread ---" = Active conversation (YOUR MEMORY)
        - If ticket exists in Current Thread: ALL requests UPDATE that ticket (unless user says "create new")
        - Find MOST RECENT ticket = active ticket
        - Only create NEW if: no ticket exists OR user says "create/new/make"

        2. MISSING INFO VALIDATION (STRICT):
        Before creating ANY ticket, you MUST have ALL of these:
        ✅ WHAT component/feature (e.g., "login button", "payment API")
        ✅ WHAT action/problem (e.g., "not working", "needs to be built", "crashes")
        ✅ Clear understanding of what needs to be done

        ❌ TOO VAGUE (DO NOT CREATE - ASK FOR DETAILS):
        - "create ticket" (no topic at all)
        - "create ticket for login" (what about login?)
        - "create ticket for login for fahad" (what needs to be done?)
        - "make a task for API" (what about the API?)
        - "add issue for dashboard" (what issue?)
        - Single word or phrase without context

        ✅ GOOD (HAS ENOUGH DETAIL - CAN CREATE):
        - "create ticket for login button not responding" (component + problem)
        - "add task to fix login bug" (action + problem)
        - "create ticket for building OAuth authentication" (action + feature)
        - "ticket for database cleanup performance issue" (component + problem)

        🚨 VALIDATION CHECKLIST (USE THIS):
        Before creating, ask yourself:
        1. Do I know WHAT component/feature this is about?
        2. Do I know WHAT needs to be done (fix/build/improve)?
        3. Do I know WHY this is needed (problem/goal)?
        
        If ANY answer is NO → ASK FOR MORE DETAILS
        
        Example Questions to Ask:
        - "What specifically about login needs attention?"
        - "What's the issue with the login? Is it broken or a new feature?"
        - "Can you describe what's happening with the login?"

        3. ASSIGNEE (CHECK MESSAGE FIRST):
        - Parse message for: "assign to [name]", "assign this to [name]", "for [name]"
        - If found: Verify name exists → Create immediately
        - If NOT found: Call get_project_assignable_users_sync → Ask "Who should work on this?"

        4. RESPONSE FORMAT:
        When create_issue_sync succeeds, return exact result["message"] - don't modify it

        5. ISSUE TYPE:
        - Default: "Story"
        - Only use "Bug" if user says "bug"

        📋 WORKFLOW

        Step 0: Detect Account → Call detect_jira_account_sync(user_query)

        Step 1: Check Existing Tickets
        - Scan Current Thread for ticket IDs (AI-123, DATA-456)
        - If found AND user NOT saying "create new": UPDATE it → Skip to Step 5
        - Create NEW only if: no ticket exists OR user says "create/new/make"

        Step 2: STRICT Validation (NEW tickets only)
        Run the validation checklist:
        ✓ Do I know the component/feature?
        ✓ Do I know what needs to be done?
        ✓ Do I understand the problem or goal?
        
        If MISSING ANY → Stop and ask clarifying questions
        If ALL CLEAR → Proceed to Step 3

        Step 3: Check Assignee (NEW tickets only)
        - Parse message for assignee patterns: "assign to [name]", "for [name]"
        - If found: Verify exists → Use directly
        - If NOT found: Call get_project_assignable_users_sync → Ask "Who should work on this?"

        Step 4: Consolidate (NEW tickets only)
        - Multiple problems → One ticket

        Step 5: Execute
        - NEW: summary, description, slack_username, channel_id, message_id
        - UPDATE: Only mentioned fields

        📝 DESCRIPTION FORMAT (NEW tickets)

        CRITICAL: Use double asterisks **text** for bold text.

        Format the description EXACTLY like this:

        **What is the request?**
        [Extract from user's message - clear description of the work]

        **Why is this important?**
        [Generate reasoning: performance impact, user experience improvement, etc.]

        **When can this ticket be closed (Definition of Done)?**
        [Include acceptance criteria if mentioned, otherwise write: "To be defined by assignee"]

        **Conversations:**
        [Include relevant context from thread if it adds value, otherwise omit this section]

        IMPORTANT: 
        - Use **text** for bold (double asterisks)
        - Add blank line between sections
        - Keep descriptions clear and actionable

        🛠️ TOOLS

        - detect_jira_account_sync(user_query)
        - create_issue_sync(assignee_email, summary, description_text, issue_type_name, slack_username, channel_id, message_id)
        - update_issue_sync(issue_key, ...)
        - get_project_assignable_users_sync()
        - get_project_epics_sync(project_key)
        - delete_issue_sync(issue_key)
        - search_confluence_knowledge_sync(user_question)

        📋 EXAMPLES

        Ex 1 - TOO VAGUE (Block):
        User: "create a ticket"
        You: "What should this ticket be about? Please describe what needs to be done."

        Ex 2 - TOO VAGUE (Block):
        User: "create ticket for login"
        You: "What specifically about login needs attention? For example, is there a bug, or do you need a new feature built?"

        Ex 3 - STILL TOO VAGUE (Block):
        User: "create ticket for login for fahad"
        You: "I understand this is for Fahad, but what specifically needs to be done with login? Is something broken, or is this a new feature?"

        Ex 4 - GOOD (Has component + problem):
        User: "create ticket for login button not responding"
        You: [detect_account] [get_users] "Who should work on this? Available: Alice, Bob, Charlie"

        Ex 5 - GOOD (Has action + clear goal):
        User: "create ticket to build OAuth authentication and assign to Bob"
        You: [detect_account] [Verify Bob exists] [Create ticket]

        Ex 6 - UPDATE:
        Thread: "Ticket created: AI-123"
        User: "add epic AI-100"
        You: [Found AI-123] [update_issue_sync(issue_key="AI-123", epic_key="AI-100")]

        🎯 KEY BEHAVIORS

        - Detect account first
        - ALWAYS validate detail level before creating
        - If only component name (e.g., "login", "API") with no action/problem → ASK
        - Parse message for assignee BEFORE asking
        - If ticket exists → UPDATE it (unless "create new")
        - Format descriptions with **text** for bold
        - Return exact tool response messages
        """

    def search_confluence_knowledge_sync(self, user_question: str) -> dict:
        """
        Search Confluence knowledge base for documentation and answers.
        LangGraph tool - searches and returns raw results for agent to process.

        Args:
            user_question: The user's question to search for in the knowledge base

        Returns:
            dict: Contains search results with documents and sources
        """
        try:
            # print("🔍 CONFLUENCE SEARCH TOOL CALLED!")  # Debug print
            # print(f"Question: {user_question}")

            # Run async search synchronously
            search_result = asyncio.run(
                self.utils.search_confluence_knowledge(user_question)
            )

            # print(f"Search result status: {search_result.get('status')}")

            if search_result["status"] == "error":
                return {
                    "success": False,
                    "query": user_question,
                    "documents": [],
                    "message": f"❌ Error: {search_result['message']}",
                }

            if not search_result["retrieved_content"]:
                return {
                    "success": False,
                    "query": user_question,
                    "documents": [],
                    "message": "I couldn't find any relevant information in the knowledge base for your question.",
                }

            # Format the documents with sources
            documents = []
            for doc in search_result["retrieved_content"]:
                documents.append(
                    {
                        "title": doc["title"],
                        "space": doc["space"],
                        "content": doc["content"],
                        "url": doc["url"],
                        "score": doc.get("score", 0),
                    }
                )

            # print(f"✅ Found {len(documents)} documents")

            # Create a formatted message with all documents and sources
            message_parts = ["📚 Found relevant documentation:\n"]

            for i, doc in enumerate(documents, 1):
                message_parts.append(
                    f"\n*Document {i}: {doc['title']}* (Space: {doc['space']})"
                )
                message_parts.append(f"Content: {doc['content'][:500]}...")
                message_parts.append(f"🔗 <{doc['url']}|View full document>")

            formatted_message = "\n".join(message_parts)

            return {
                "success": True,
                "query": user_question,
                "documents": documents,
                "message": formatted_message,
                "total_found": len(documents),
            }

        except Exception as e:
            logger.error(f"Error in Confluence search: {e}")
            print(f"❌ ERROR: {e}")
            return {
                "success": False,
                "query": user_question,
                "documents": [],
                "message": f"❌ Error searching knowledge base: {str(e)}",
            }

    def create_issue_sync(
        self,
        project_name_or_key: str = "",
        summary: str = "",
        description_text: str = "",
        assignee_email: str = "",
        priority_name: str = None,
        reporter_email: str = None,
        issue_type_name: str = None,
        sprint_name: str = None,
        story_points: int = None,
        epic_key: str = None,
        slack_username: str = None,
        channel_id: str = None,
        message_id: str = None,
        force_update_description_after_create: bool = True,
    ) -> dict:
        """
        Create a new Jira ticket/issue.

        🔴 ASSIGNEE IS MANDATORY 🔴
        This function requires an assignee. If no assignee is provided, you must ask the user.

        ⚠️ CRITICAL RESPONSE HANDLING ⚠️
        When this function succeeds, it returns a pre-formatted message in result["message"].
        YOU MUST return this message EXACTLY as-is. Do not add commentary or modify it.

        The message format is:
        Ticket created: [URL]
        Assigned to: [NAME]
        Epic: [KEY or "unknown - would you like a list of epics to choose from?"]

        Just return result["message"] directly - nothing else!

        Args:
            assignee_email: REQUIRED - Who to assign ticket to
            summary: Ticket title
            description_text: Ticket description
            slack_username: Slack username for reporter matching
            channel_id: Slack channel ID (for thread link)
            message_id: Slack message timestamp (for thread link)
            ... (other args)

        Returns:
            dict: Contains "success" (bool) and "message" (str - pre-formatted response)

        Examples:
            result = create_issue_sync(assignee_email="john", summary="Fix bug", channel_id="C123", message_id="1234.567")
            # Return result["message"] directly!
        """
        if not project_name_or_key or not project_name_or_key.strip():
            current_config = Utils.get_account_config(self.utils.current_account)
            project_name_or_key = current_config.get(
                "project_key", os.getenv("Default_Project")
            )
            logger.info(
                f"📌 No project specified, using current account's project: {project_name_or_key}"
            )

        return self.utils.create_issue_implementation(
            project_name_or_key,
            summary,
            description_text,
            assignee_email,
            priority_name,
            reporter_email,
            issue_type_name,
            sprint_name,
            story_points,
            epic_key,
            slack_username,
            channel_id,
            message_id,
        )

    def update_issue_sync(
        self,
        issue_key: str,
        summary: str = None,
        description_text: str = None,
        assignee_email: str = None,
        priority_name: str = None,
        due_date: str = None,
        issue_type_name: str = None,
        labels: str = None,
        sprint_name: str = None,
        status_name: str = None,
        story_points: int = None,
        epic_key: str = None,
    ) -> dict:
        """
        Update an existing Jira ticket by issue key.

        Args:
            issue_key: REQUIRED - The ticket ID (PROJECT-123 format)
            summary: New title/summary
            description_text: New description content
            assignee_email: New assignee email or "unassigned"
            priority_name: New priority (High/Medium/Low)
            due_date: Due date (YYYY-MM-DD format)
            issue_type_name: New issue type
            labels: Comma-separated labels
            sprint_name: Sprint name or "backlog"
            status_name: New status (In Progress/Done/To Do)
            story_points: Story points (1, 2, 3, 5, 8, 13, etc.)
            epic_key: Epic key to link this ticket to (e.g., "AI-123")

        Returns:
            dict: Updated ticket information

        Examples:
            ✅ update_issue_sync(issue_key="AI-123", summary="New title")
            ✅ update_issue_sync(issue_key="PROJ-456", assignee_email="john", story_points=8)
        """
        return self.utils.update_issue(
            issue_key,
            summary,
            description_text,
            assignee_email,
            priority_name,
            due_date,
            None,
            issue_type_name,
            labels.split(",") if labels else None,
            sprint_name,
            status_name,
            story_points,
            epic_key,
        )

    def get_project_assignable_users_sync(self, project_key: str = None) -> dict:
        """
        Get list of users who can be assigned tickets in a project.
        Simple tool for showing available assignees to users.

        Args:
            project_key: Project key (if None, uses current account's project)

        Returns:
            dict: List of assignable users with their display names
        """
        try:
            # ✅ If no project specified, use current account's project
            if not project_key:
                current_config = Utils.get_account_config(self.utils.current_account)
                project_key = current_config.get(
                    "project_key", os.getenv("Default_Project")
                )
                logger.info(f"📌 Using current account's project: {project_key}")

            url = f"{self.utils.base_url}/rest/api/3/user/assignable/search"
            params = {"project": project_key, "maxResults": 20}

            logger.info(
                f"Fetching users for project: {project_key} from account: {self.utils.current_account}"
            )
            logger.info(f"Using base URL: {self.utils.base_url}")

            response = self.utils.session.get(url, params=params, timeout=30)

            if response.status_code == 200:
                users_data = response.json()

                # Simple list of display names
                user_names = []
                for user in users_data:
                    if user.get("active", True):
                        display_name = user.get("displayName", "")
                        if display_name:
                            user_names.append(display_name)

                user_names.sort()  # Alphabetical order

                return {
                    "success": True,
                    "project": project_key,
                    "account": self.utils.current_account,
                    "users": user_names,
                    "formatted_list": "\n".join([f"• {name}" for name in user_names]),
                }
            else:
                logger.error(f"Failed to fetch users. Status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return {
                    "success": False,
                    "error": f"Could not fetch users for project {project_key}",
                    "users": [],
                }

        except Exception as e:
            logger.error(f"Error fetching project users: {e}")
            return {"success": False, "error": str(e), "users": []}

    def get_project_epics_sync(self, project_key: str = None) -> dict:
        """
        Get list of epics for a project to allow users to link tickets to epics.

        Args:
            project_key: Project key (if None, uses current account's project)

        Returns:
            dict: List of epics with their keys and summaries
        """
        try:
            # ✅ If no project specified, use current account's project
            if not project_key:
                current_config = Utils.get_account_config(self.utils.current_account)
                project_key = current_config.get(
                    "project_key", os.getenv("Default_Project")
                )
                logger.info(f"📌 Using current account's project: {project_key}")

            result = self.utils.get_project_epics_implementation(project_key)

            # Enhanced error handling with better user guidance
            if not result["success"]:
                if "jira library" in result.get("error", "").lower():
                    result["user_message"] = (
                        f"Epic discovery requires the 'jira' Python library to be installed. "
                        f"If you know a specific epic key (like {project_key}-100), I can still link to it directly."
                    )
                else:
                    result["user_message"] = (
                        f"Unable to fetch epics right now. If you have a specific epic key in mind, "
                        f"please provide it and I'll link the ticket to it."
                    )

            # Add helpful guidance when no epics are found
            elif result["success"] and not result.get("epics"):
                result["user_message"] = (
                    f"No epics found in {project_key}. You can create epics in Jira first, "
                    f"or if you have an existing epic key, I can link to it directly."
                )

            return result

        except Exception as e:
            logger.error(f"Error in get_project_epics_sync: {e}")
            return {
                "success": False,
                "project": project_key,
                "epics": [],
                "error": str(e),
            }

    def delete_issue_sync(self, issue_key: str) -> dict:
        """
        Delete a Jira ticket permanently.

        Args:
            issue_key: REQUIRED - The ticket ID to delete (PROJECT-123 format)

        Returns:
            dict: Success status and deleted ticket info

        Examples:
            ✅ delete_issue_sync(issue_key="AI-123")
        """
        return self.utils.delete_issue(issue_key)

    def get_sprint_list_sync(self, project_name_or_key: str) -> dict:
        """Get available sprints for a project."""
        if not project_name_or_key:
            current_config = Utils.get_account_config(self.utils.current_account)
            project_name_or_key = current_config.get(
                "project_key", os.getenv("Default_Project")
            )
            logger.info(
                f"📌 No project specified for sprints, using current account's project: {project_name_or_key}"
            )

        return self.utils.get_sprint_list_implementation(project_name_or_key)

    def get_project_from_issue_sync(self, issue_key: str) -> dict:
        """Get project key from issue key."""
        return self.utils.get_project_from_issue_implementation(issue_key)

    async def refactor_query_with_context(
        self, original_query: str, chat_history: str
    ) -> str:
        """Refactor user query by incorporating relevant context from chat history."""
        try:
            logger.info("=== QUERY REFINEMENT START ===")

            # Skip if no chat history or very short queries
            if (
                not chat_history
                or not chat_history.strip()
                or len(original_query.split()) <= 2
            ):
                logger.info("Skipping refinement: insufficient context")
                return original_query

            # Simple prompt for AI to understand context
            prompt = f"""Fix the grammar and spelling of this user request. Only correct grammar, spelling, and basic sentence structure. Do not add any context, assignees, or change the meaning.

USER REQUEST: {original_query}

Return only the grammatically corrected request:"""

            # Call OpenAI directly
            try:
                import openai

                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0,
                )

                if response and response.choices and len(response.choices) > 0:
                    refined_query = response.choices[0].message.content.strip()

                    # Don't use if response is way too long
                    if len(refined_query) > len(original_query) * 3:
                        logger.warning("Refined query too long, using original")
                        return original_query

                    logger.info(
                        f"Query refined: '{original_query}' → '{refined_query}'"
                    )
                    return refined_query
                else:
                    logger.info("No refinement response, using original")
                    return original_query

            except Exception as api_error:
                logger.error(f"OpenAI API error: {api_error}")
                return original_query

        except Exception as e:
            logger.error(f"Refinement error: {e}")
            return original_query

    async def process_query(
        self,
        user_query: UserQuery,
        channel_id: str = None,
        message_id: str = None,
        slack_username: str = None,
    ) -> dict:
        """Process Jira query using single agent with Slack tracking."""
        try:
            logger.info(f"Processing query: {user_query.query}")
            logger.info(f"Channel ID: {channel_id}")
            logger.info(f"Message ID: {message_id}")

            # Get raw chat history - no interpretation
            chat_history_string = self.utils.extract_chat(channel_id, message_id)
            # Query refinement functionality
            refined_query = await self.refactor_query_with_context(
                user_query.query, chat_history_string
            )
            logger.info(f"Original query: {user_query.query}")
            logger.info(f"Refined query: {refined_query}")
            # Give refined query and context to the agent
            content = f"""
            USER QUERY: {refined_query}

            AVAILABLE JIRA ACCOUNTS:
            {chr(10).join(f"- {k}: {v['name']}" for k, v in JIRA_ACCOUNTS.items())}

            ORIGINAL QUERY: {user_query.query}

            SLACK USERNAME: {slack_username if slack_username else "Not provided"}

            SLACK CONTEXT (for adding thread link to Jira description):
            - Channel ID: {channel_id if channel_id else "Not provided"}
            - Message ID: {message_id if message_id else "Not provided"}

            CONVERSATION HISTORY: 
            {chat_history_string}

            INSTRUCTIONS:
            1. FIRST: Call detect_jira_account_sync to identify which account to use
            2. The account detection will automatically switch credentials
            3. Then proceed with normal Jira operations using the detected account
            4. Always confirm which account you're using when creating/updating tickets

            - The refined query above has been enhanced with context from the conversation history
            - Use the refined query as your primary instruction, but refer to original and history for additional context
            - IMPORTANT: When creating tickets, ALWAYS pass these parameters to create_issue_sync:
            * slack_username (for reporter matching)
            * channel_id (for Slack thread link)
            * message_id (for Slack thread link)
            - The Slack thread link will be automatically added to the Jira ticket description
            - Extract any mentioned assignees, issue keys, priorities, etc. from all sources
            - Call the appropriate tool based on your analysis of the refined query
            """

            result = self.jira_agent.invoke(
                {"messages": [{"role": "user", "content": content}]}
            )

            if result and "messages" in result and len(result["messages"]) > 0:
                final_message = result["messages"][-1]
                response_content = final_message.content
                formatted_response = self.utils.format_for_slack(response_content)

                # Extract issue key from response
                issue_key = self.extract_issue_key_from_response(formatted_response)
                logger.info(f"Extracted issue key: {issue_key}")

                # Check if this is a creation response - EXPANDED KEYWORDS for multi-step
                response_lower = formatted_response.lower()
                creation_keywords = [
                    "successfully created",
                    "created the jira issue",
                    "created the story",
                    "created the task",
                    "created the bug",
                    "created the epic",
                    "i have successfully created",
                    "i've created the",
                    "created for you",
                    "new jira issue",
                    "new ticket created",
                    "issue has been created",
                    "ticket created",
                    "created the ticket",
                    "ticket for you:",
                    "assigned to",
                    "created the tickets",
                ]

                is_creation = any(
                    keyword in response_lower for keyword in creation_keywords
                )
                logger.info(f"Is creation: {is_creation}")

                # TRACKING LOGIC - handle both direct and multi-step scenarios
                if is_creation and channel_id and issue_key:
                    logger.info("This is a creation response - handling tracking")

                    try:
                        # Determine if this is from slash command or regular message
                        is_slash_command = message_id == "SLASH_COMMAND"

                        if is_slash_command:
                            # For slash commands, find the user trigger
                            user_trigger_timestamp = self.find_recent_user_trigger(
                                channel_id, user_query.query
                            )
                        else:
                            # For regular messages, use the message_id we received
                            user_trigger_timestamp = message_id

                        if user_trigger_timestamp:
                            logger.info(
                                f"Using trigger timestamp: {user_trigger_timestamp}"
                            )

                            # Save tracking data
                            channel_name = self.utils.get_channel_name(channel_id)
                            self.utils.save_slack_tracking_data(
                                message_id=user_trigger_timestamp,
                                channel_id=channel_id,
                                channel_name=channel_name,
                                issue_key=issue_key,
                            )
                            logger.info(
                                f"✅ Successfully saved tracking data for issue {issue_key}"
                            )
                        else:
                            logger.warning("Could not find user trigger timestamp")

                    except Exception as track_error:
                        logger.error(f"Error with tracking: {track_error}")

                return {
                    "success": True,
                    "message": "Jira operation completed",
                    "data": formatted_response,
                    "query": user_query.query,
                    "refined_query": refined_query,
                    "issue_key": issue_key,
                }
            else:
                return {
                    "success": False,
                    "message": "No response generated from Jira processing",
                    "data": None,
                    "query": user_query.query,
                }

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to process Jira query: {str(e)}",
                "data": None,
                "query": user_query.query,
            }

    def find_recent_user_trigger(self, channel_id: str, original_query: str) -> str:
        """Find the most recent user message that could be the trigger"""
        try:
            from slack_sdk import WebClient

            client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

            # Get recent message history
            response = client.conversations_history(
                channel=channel_id, limit=20, inclusive=True
            )

            if not response["ok"]:
                logger.error(f"Failed to get message history: {response['error']}")
                return None

            messages = response["messages"]
            messages.sort(key=lambda x: float(x["ts"]), reverse=True)  # Newest first

            # Look for recent user messages (not bot messages)
            for msg in messages:
                # Skip bot messages
                if msg.get("bot_id") or msg.get("user") in ["bot", None]:
                    continue

                text = msg.get("text", "").lower()

                # Look for slash commands or related content
                if text.startswith("/jira") or any(
                    word in text
                    for word in ["jira", "ticket", "create", "assign"]
                    if len(word) > 3
                ):
                    logger.info(f"Found recent user trigger: {msg['ts']} - {text[:50]}")
                    return msg["ts"]

            # Fallback: return the most recent user message
            for msg in messages:
                if not msg.get("bot_id") and msg.get("user") not in ["bot", None]:
                    logger.info(
                        f"Using most recent user message as trigger: {msg['ts']}"
                    )
                    return msg["ts"]

            logger.warning("Could not find any user trigger message")
            return None

        except Exception as e:
            logger.error(f"Error finding recent user trigger: {e}")
            return None
