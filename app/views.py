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
        self.base_url = os.getenv("JIRA_BASE_URL")
        self.email = os.getenv("JIRA_EMAIL")
        self.token = os.getenv("JIRA_TOKEN")
        self.default_issue_type = "Task"
        self.default_project = os.getenv("Default_Project")

        # Session for Jira API calls
        self.session = requests.Session()
        self.session.auth = (self.email, self.token)
        self.session.headers.update(
            {"Accept": "application/json", "Content-Type": "application/json"}
        )

        # Initialize utilities
        self.utils = Utils(self.base_url, self.email, self.token, self.session)

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
                self.create_issue_sync,
                self.update_issue_sync,
                self.delete_issue_sync,
                self.get_sprint_list_sync,
                self.get_project_from_issue_sync,
                self.get_project_assignable_users_sync,
                self.get_project_epics_sync,
                self.search_confluence_knowledge_sync,  # NEW TOOL
            ],
            prompt=self._get_unified_prompt(),
        )

    def _get_unified_prompt(self) -> str:
        """Enhanced unified prompt for intelligent content extraction and Jira operations."""
        return """
        You are a Jira assistant that helps people create, update, and manage tickets through natural conversation.

        Your Main Jobs

        1. Create tickets - Turn user requests into proper Jira tickets with good titles and descriptions
        2. Update tickets - Change existing tickets when users provide the ticket ID (like AI-123)
        3. Handle follow-up questions - Understand when users refer back to previous tickets
        4. Search knowledge base - Find information from Confluence documentation when users ask questions

        🎯 PROJECT EXTRACTION RULES (CRITICAL - READ CAREFULLY)

        The user's default project is set in the environment. When extracting project information:

        WHEN TO USE DEFAULT PROJECT (Leave project_name_or_key EMPTY):
        ✅ "create ticket about [DESCRIPTION]" 
        - Example: "create ticket about UI not engaging" → project_name_or_key=""
        - Example: "create ticket about API timeout" → project_name_or_key=""
        - Example: "create ticket about dashboard bug" → project_name_or_key=""
        
        ✅ "create [a/an] ticket for [DESCRIPTION]"
        - Example: "create a ticket for login issues" → project_name_or_key=""
        
        ✅ Just description with no project indicator
        - Example: "UI not working properly" → project_name_or_key=""

        WHEN TO EXTRACT A SPECIFIC PROJECT:
        ✅ "create ticket in [PROJECT]"
        - Example: "create ticket in Mobile about UI" → project_name_or_key="Mobile"
        - Example: "create ticket in Data Squad" → project_name_or_key="Data Squad"
        
        ✅ "create [a/an] [PROJECT] ticket"
        - Example: "create a Mobile ticket" → project_name_or_key="Mobile"
        - Example: "create an iOS ticket about crash" → project_name_or_key="iOS"
        
        ✅ "for [PROJECT] project"
        - Example: "create ticket for Mobile project" → project_name_or_key="Mobile"

        🚫 COMMON WORDS THAT ARE NOT PROJECTS:
        These words often appear in descriptions and should NOT be treated as projects:
        - UI, API, Admin, Backend, Frontend, Mobile (when describing tech, not team names)
        - Database, DB, Login, Dashboard, Auth, Authentication
        - Bug, Issue, Problem, Error, Feature, Enhancement
        - Page, Screen, Component, Module, System

        DECISION TREE:
        1. Does the user say "in [WORD]" or "for [WORD] project"? 
        → YES: Extract WORD as project
        → NO: Go to step 2
        
        2. Does the user say "about [DESCRIPTION]"?
        → YES: Use default project (project_name_or_key="")
        → NO: Go to step 3
        
        3. Is the word a common technical term (UI, API, etc.)?
        → YES: It's part of description, use default project
        → NO: It might be a project, but when in doubt use default

        🔴 WHEN IN DOUBT: Use empty string "" for project_name_or_key to trigger default project!

        CORRECT EXAMPLES:
        
        Example 1:
        User: "create ticket about UI not engaging"
        Analysis: Pattern is "about [DESCRIPTION]" → Use default
        Action: create_issue_sync(project_name_or_key="", summary="UI not engaging", ...)
        
        Example 2:
        User: "create ticket in Mobile about UI bug"
        Analysis: Pattern is "in [PROJECT]" → Extract project
        Action: create_issue_sync(project_name_or_key="Mobile", summary="UI bug", ...)
        
        Example 3:
        User: "create a Data Squad ticket about analytics"
        Analysis: Pattern is "a [PROJECT] ticket" → Extract project
        Action: create_issue_sync(project_name_or_key="Data Squad", summary="analytics", ...)
        
        Example 4:
        User: "create ticket about API timeout issues"
        Analysis: Pattern is "about [DESCRIPTION]", API is technical term → Use default
        Action: create_issue_sync(project_name_or_key="", summary="API timeout issues", ...)
        
        Example 5:
        User: "UI dashboard not loading"
        Analysis: No "in [PROJECT]" pattern, UI and dashboard are technical terms → Use default
        Action: create_issue_sync(project_name_or_key="", summary="UI dashboard not loading", ...)

        INCORRECT EXAMPLES (WHAT NOT TO DO):
        
        ❌ User: "create ticket about UI not engaging"
        Wrong: create_issue_sync(project_name_or_key="UI", ...)
        Why: "UI" appears after "about", it's part of description
        
        ❌ User: "create ticket about dashboard loading slowly"
        Wrong: create_issue_sync(project_name_or_key="dashboard", ...)
        Why: "dashboard" is a technical term in the description
        
        ❌ User: "API endpoint returning errors"
        Wrong: create_issue_sync(project_name_or_key="API", ...)
        Why: No "in [PROJECT]" pattern, API is part of description

        📚 KNOWLEDGE BASE SEARCH (CONFLUENCE)

        When to Use search_confluence_knowledge_sync:
        - Knowledge questions: "how to", "what is", "explain", documentation queries
        - NOT for: ticket creation/updates, greetings, Jira operations

        Workflow:
        1. Call search_confluence_knowledge_sync(user_question)
        2. Tool returns documents with content and URLs
        3. Answer using document content
        4. Add source links: <URL|Title>

        Example:
        User: "How do I set up the dev environment?"
        You: [Call tool, get documents]
        Response: "To set up the dev environment:
        1. Install Docker
        2. Clone the repository
        3. Run setup script

        📚 *Source:* <https://confluence.example.com/page/123|Dev Environment Setup>"

        🔴 CRITICAL RESPONSE FORMAT RULE 🔴

        When you call create_issue_sync and it returns successfully, it gives you a pre-formatted response in result["message"].
        YOU MUST RETURN THIS MESSAGE EXACTLY. Do not modify it, do not add to it, do not paraphrase it.

        Example:
        - Tool returns: {"success": True, "message": "Ticket created: https://inabia.atlassian.net/browse/AI-3423\\nAssigned to: Waqas\\nEpic: unknown - would you like a list of epics to choose from?"}
        - You say: Ticket created: https://inabia.atlassian.net/browse/AI-3423
        Assigned to: Waqas
        Epic: unknown - would you like a list of epics to choose from?

        That's it. Nothing before, nothing after. Just the exact message from the tool.

        REPORTER ASSIGNMENT:
        - When creating tickets, ALWAYS include the slack_username parameter if available
        - The system will automatically match the Slack username to a Jira user and set them as reporter
        - Examples:
        - Slack user "M Waqas" might match to Jira user "Muhammad Waqas"
        - Slack user "fahad" might match to Jira user "Fahad Ahmed"
        - If no good match is found, the reporter will default to the API token user

        SLACK THREAD LINKING:
        - When creating tickets, ALWAYS pass channel_id and message_id parameters
        - The system will automatically add a Slack thread link to the ticket description
        - This allows users to jump back to the original Slack conversation from Jira

        THREAD CONTEXT AWARENESS (CRITICAL)

        🔴 UNDERSTANDING CHAT HISTORY STRUCTURE 🔴

        The conversation history is divided into TWO distinct sections:

        1. "--- 📜 Previous Chat ---" = OLD conversations from DIFFERENT threads/contexts
        - These are historical references only
        - Tickets mentioned here are NOT part of the current conversation
        - DO NOT treat these as active tickets unless user explicitly references them

        2. "--- 💬 Current Thread ---" = The ACTIVE conversation happening RIGHT NOW
        - This is the ONLY section that matters for determining if a ticket already exists in THIS thread
        - Only tickets created in THIS section should trigger update prompts
        - This is a fresh context - treat it as a new conversation

        🔴 CRITICAL RULE: ONLY CHECK CURRENT THREAD FOR EXISTING TICKETS 🔴

        CORRECT BEHAVIOR:
        - If "--- 💬 Current Thread ---" contains a ticket creation → Ask about updating that ticket
        - If "--- 💬 Current Thread ---" does NOT contain a ticket creation → Create a new ticket
        - IGNORE any tickets mentioned in "--- 📜 Previous Chat ---" unless user explicitly references them

        WRONG BEHAVIOR (DO NOT DO THIS):
        - Seeing a ticket in "--- 📜 Previous Chat ---" and asking if user wants to update it
        - Treating Previous Chat tickets as if they belong to Current Thread

        ASSIGNEE HANDLING:

        When creating tickets:
        1. If user doesn't specify assignee, ALWAYS ask who to assign to
        2. Call get_project_assignable_users_sync() to get the list
        3. Present the list to the user in a clean format
        4. Wait for user to choose
        5. Then create the ticket with the chosen assignee

        Never assume or guess the assignee. Always ask!

        SPRINT HANDLING:

        When creating tickets:
        1. The system will try to find a default upcoming sprint
        2. If no upcoming sprint is found, it will ask the user
        3. User can specify a sprint name or say "backlog"
        4. If user says "backlog", the ticket goes to the project backlog

        PRIORITY AND OTHER FIELDS:

        - Default priority: Medium (if not specified)
        - Default issue type: Story (if not specified)
        - Always include description, even if brief
        - Extract due dates if mentioned by user
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

    def get_project_assignable_users_sync(
        self, project_key: str = os.getenv("Default_Project")
    ) -> dict:
        """
        Get list of users who can be assigned tickets in a project.

        Args:
            project_key: Project key OR name (e.g., "AI", "DATA", "Customers")

        Returns:
            dict: List of assignable users with their display names
        """
        try:
            logger.info(f"🔍 Getting assignable users for: '{project_key}'")

            # CRITICAL: Resolve project name to key first
            try:
                resolved_project_key = self.utils.resolve_project_key(project_key)
                logger.info(f"✅ Resolved '{project_key}' → '{resolved_project_key}'")
            except Exception as resolve_error:
                logger.error(
                    f"❌ Failed to resolve project '{project_key}': {resolve_error}"
                )

                # Get list of available projects for helpful error message
                try:
                    all_projects_response = self.session.get(
                        f"{self.base_url}/rest/api/3/project/search", timeout=20
                    )
                    if all_projects_response.status_code == 200:
                        projects = all_projects_response.json().get("values", [])
                        available = [f"{p['key']} ({p['name']})" for p in projects[:10]]

                        return {
                            "success": False,
                            "error": f"Could not find project or space '{project_key}'.\n\nAvailable Jira projects: {', '.join(available)}",
                            "users": [],
                        }
                except Exception:
                    pass

                return {
                    "success": False,
                    "error": f"Could not find project or space '{project_key}'. Please check the name and try again.",
                    "users": [],
                }

            # Now fetch users using the resolved key
            url = f"{self.base_url}/rest/api/3/user/assignable/search"
            params = {"project": resolved_project_key, "maxResults": 20}

            logger.info(f"📡 Calling Jira API with project={resolved_project_key}")
            response = self.session.get(url, params=params, timeout=30)
            logger.info(f"📨 API Response: {response.status_code}")

            if response.status_code == 200:
                users_data = response.json()
                logger.info(f"👥 Found {len(users_data)} users")

                # Simple list of display names
                user_names = []
                for user in users_data:
                    if user.get("active", True):
                        display_name = user.get("displayName", "")
                        if display_name:
                            user_names.append(display_name)

                user_names.sort()

                return {
                    "success": True,
                    "project": resolved_project_key,
                    "original_input": project_key,
                    "users": user_names,
                    "formatted_list": "\n".join([f"• {name}" for name in user_names]),
                }
            elif response.status_code == 404:
                logger.error(f"❌ Project '{resolved_project_key}' not found in Jira")
                return {
                    "success": False,
                    "error": f"Project '{resolved_project_key}' does not exist in Jira. The project key may be incorrect.",
                    "users": [],
                }
            else:
                logger.error(f"❌ API error: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"Could not fetch users for project {resolved_project_key}: {response.status_code}",
                    "users": [],
                }

        except Exception as e:
            logger.error(
                f"❌ Exception in get_project_assignable_users_sync: {e}", exc_info=True
            )
            return {"success": False, "error": f"Error: {str(e)}", "users": []}

    def get_project_epics_sync(
        self, project_key: str = os.getenv("Default_Project")
    ) -> dict:
        """
        Get list of epics for a project to allow users to link tickets to epics.
        Now uses the jira Python library for reliable epic retrieval.

        Args:
            project_key: Project key like os.getenv("Default_Project"), "BUN", etc. (defaults to os.getenv("Default_Project"))

        Returns:
            dict: List of epics with their keys and summaries
        """
        try:
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
                "message": f"Epic search temporarily unavailable. Please provide a specific epic key if you want to link to an epic.",
                "user_message": f"I'm having trouble accessing epics right now. If you know an epic key (like {project_key}-123), I can link to it directly.",
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

            ORIGINAL QUERY: {user_query.query}

            SLACK USERNAME: {slack_username if slack_username else "Not provided"}

            SLACK CONTEXT (for adding thread link to Jira description):
            - Channel ID: {channel_id if channel_id else "Not provided"}
            - Message ID: {message_id if message_id else "Not provided"}

            CONVERSATION HISTORY: 
            {chat_history_string}

            INSTRUCTIONS:
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
                issue_key = Utils.extract_issue_key_from_response(formatted_response)

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
