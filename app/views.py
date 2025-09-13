import os
import requests
import logging
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from .schemas import UserQuery
from .utilities.utils import Utils
from dotenv import load_dotenv
from fastapi.responses import PlainTextResponse
from fastapi import Request


load_dotenv()
logger = logging.getLogger(__name__)


class JiraService:
    """Simplified Jira service with single React agent for all CRUD operations."""

    def __init__(self):
        """Initialize Jira service with single agent and all tools."""
        self.model = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        self.base_url = os.getenv("JIRA_BASE_URL")
        self.email = os.getenv("JIRA_EMAIL")
        self.token = os.getenv("JIRA_TOKEN")
        self.default_issue_type = "Task"
        self.default_project = "AI"

        # Session for Jira API calls
        self.session = requests.Session()
        self.session.auth = (self.email, self.token)
        self.session.headers.update(
            {"Accept": "application/json", "Content-Type": "application/json"}
        )

        # Initialize utilities
        self.utils = Utils(self.base_url, self.email, self.token, self.session)

        # Create single agent with all CRUD tools
        self.jira_agent = create_react_agent(
            model=self.model,
            tools=[
                self.create_issue_sync,
                self.update_issue_sync,
                self.delete_issue_sync,
                self.get_sprint_list_sync,
                self.get_project_from_issue_sync,
                self.get_project_assignable_users_sync,
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

Creating New Tickets

When someone asks you to create a ticket:

Extract the important details:
- What type of work is it? (story, task, bug, epic)
- What's it about? (make a clear title from their request)
- Who should work on it?

If you have everything needed: Create the ticket right away

If you're missing the assignee: Ask who should work on it. First call get_project_assignable_users_sync to show them available people, then ask them to choose.

Good examples:
- "create story for API testing" → You ask: "Who should work on this? Here are the available people: [list users]"
- "create bug for login issue assign to john" → You create it immediately with assignee="john"

CRITICAL: Recognizing Assignee Responses

If you just asked "who should I assign this to?" and showed a user list, then the user responds with ANY of these patterns, they are giving you the assignee name:

- Just a name: "fahad" → assignee="fahad"
- Slash command with name: "/jiratest fahad" → assignee="fahad"  
- Slash command with name: "/jira john" → assignee="john"
- With assign word: "assign to sarah" → assignee="sarah"
- Simple response: "mike" → assignee="mike"

IMPORTANT: If the name they give matches someone from the user list you just showed, immediately create the ticket with that person as assignee. DO NOT ask for assignee again.

What to do when they respond with a name:
1. Check if that name was in the user list you just showed
2. If yes, create the ticket immediately using that assignee
3. If no, ask them to pick someone from the available list

Making Good Ticket Content

Write clear summaries:
- "Fix login authentication bug" ✓
- "Database migration story" ✓
- "New ticket" ✗ (too generic)

Write helpful descriptions:
- For bugs: explain what's broken and the impact
- For stories: explain what needs to be investigated or built
- For tasks: describe the work that needs doing

Understanding Context

Pay attention to how people refer to things:
- "assign it to sarah" = assign the ticket we just talked about to sarah
- "update that ticket" = update the most recent ticket mentioned
- "move AI-123 to done" = update ticket AI-123 status to done

Remember what happened in the conversation:
- If you asked for assignee and showed user list, expect their next response to be picking someone
- Keep track of what ticket you were creating when you asked for assignee

When to Use Each Tool

create_issue_sync - When someone wants a new ticket
- Need: assignee, summary, type
- Example: create_issue_sync(assignee_email="john", summary="Fix login bug", issue_type_name="Bug")

update_issue_sync - When someone wants to change an existing ticket  
- Need: ticket ID (like AI-123)
- Example: update_issue_sync(issue_key="AI-123", assignee_email="sarah")

get_project_assignable_users_sync - When you need to show who can be assigned tickets
- Use this when asking for assignees
- Shows a nice list of available people

delete_issue_sync - When someone wants to delete a ticket
- Need: ticket ID
- Example: delete_issue_sync(issue_key="AI-123")

Response Style

Be conversational and helpful:
- "I've created the story for you: AI-456"
- "I've updated the ticket and assigned it to Sarah"
- "I need to know who should work on this. Here are your options..."

Don't be robotic:
- "Ticket creation successful" ✗
- "Operation completed" ✗

Important Rules

1. Never create tickets without assignees - Always ask if you don't know
2. Always show available users when asking for assignees
3. Use the exact names people give you - don't expand "john" to "John Smith"
4. Make meaningful summaries - not generic ones
5. Remember the conversation - understand when they refer back to previous tickets
6. If you showed user list and they pick a name from it, create the ticket immediately - don't ask again

Example Conversations

Scenario 1 - Complete Flow:
User: "create story for database cleanup"
You: Call get_project_assignable_users_sync, then say: "Who should work on this database cleanup story? Available people: [user list]. Please let me know who should handle it."
User: "fahad" (or "/jiratest fahad")
You: Create ticket immediately with assignee="fahad"

Scenario 2 - Direct Assignment:
User: "create bug for login issue assign to john"
You: Call create_issue_sync immediately with assignee="john"

Scenario 3 - Updates:
User: "update AI-123 priority to high"
You: Call update_issue_sync(issue_key="AI-123", priority_name="High")

Your goal is to make Jira operations feel natural and easy for users while ensuring all tickets are properly created with the right information

    """

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
        force_update_description_after_create: bool = True,
    ) -> dict:
        """
        Create a new Jira ticket/issue.

        🔴 **ASSIGNEE IS MANDATORY** 🔴
        This function requires an assignee. If no assignee is provided, you must ask the user.

        Args:
            assignee_email: REQUIRED - Who to assign ticket to (cannot be empty)
            project_name_or_key: Project key (defaults to "AI")
            summary: Ticket title (auto-generated if empty)
            description_text: Ticket description (auto-generated if empty)
            priority_name: Priority level (High/Medium/Low)
            reporter_email: Who reported it (optional)
            issue_type_name: Issue type (Task/Story/Bug/Epic)
            sprint_name: Sprint name or "backlog"

        Returns:
            dict: Success with ticket details OR error if assignee missing

        Examples:
            ✅ create_issue_sync(assignee_email="john", summary="Fix login bug")
            ❌ create_issue_sync(summary="Fix bug") # Will ask for assignee
        """
        # Implementation from your existing code
        return self.utils.create_issue_implementation(
            project_name_or_key,
            summary,
            description_text,
            assignee_email,
            priority_name,
            reporter_email,
            issue_type_name,
            sprint_name,
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

        Returns:
            dict: Updated ticket information

        Examples:
            ✅ update_issue_sync(issue_key="AI-123", summary="New title")
            ✅ update_issue_sync(issue_key="PROJ-456", assignee_email="john")
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
        )

    def get_project_assignable_users_sync(self, project_key: str = "AI") -> dict:
        """
        Get list of users who can be assigned tickets in a project.
        Simple tool for showing available assignees to users.

        Args:
            project_key: Project key like "AI", "BUN", etc. (defaults to "AI")

        Returns:
            dict: List of assignable users with their display names
        """
        try:
            url = f"{self.base_url}/rest/api/3/user/assignable/search"
            params = {"project": project_key, "maxResults": 20}

            response = self.session.get(url, params=params, timeout=30)

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
                    "users": user_names,
                    "formatted_list": "\n".join([f"• {name}" for name in user_names]),
                }
            else:
                return {
                    "success": False,
                    "error": f"Could not fetch users for project {project_key}",
                    "users": [],
                }

        except Exception as e:
            logger.error(f"Error fetching project users: {e}")
            return {"success": False, "error": str(e), "users": []}

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
        self, user_query: UserQuery, channel_id: str = None, message_id: str = None
    ) -> dict:
        """Process Jira query using single agent with Slack tracking."""
        try:
            logger.info(f"Processing query: {user_query.query}")
            logger.info(f"Channel ID: {channel_id}")

            # Get raw chat history - no interpretation
            chat_history_string = self.utils.extract_chat(channel_id)

            # KEEP THIS - Important query refinement functionality
            refined_query = await self.refactor_query_with_context(
                user_query.query, chat_history_string
            )
            logger.info(f"Original query: {user_query.query}")
            logger.info(f"Refined query: {refined_query}")

            # Give refined query and context to the agent
            content = f"""
    USER QUERY: {refined_query}

    ORIGINAL QUERY: {user_query.query}

    CONVERSATION HISTORY: 
    {chat_history_string}

    INSTRUCTIONS:
    - The refined query above has been enhanced with context from the conversation history
    - Use the refined query as your primary instruction, but refer to original and history for additional context
    - Extract any mentioned assignees, issue keys, priorities, etc. from all sources
    - Call the appropriate tool based on your analysis of the refined query
    """

            # Execute with single agent using refined query
            result = self.jira_agent.invoke(
                {"messages": [{"role": "user", "content": content}]}
            )

            if result and "messages" in result and len(result["messages"]) > 0:
                final_message = result["messages"][-1]
                response_content = final_message.content
                formatted_response = self.utils.format_for_slack(response_content)

                # Extract issue key from response
                issue_key = self.utils.extract_issue_key_from_response(
                    formatted_response
                )
                logger.info(f"Extracted issue key: {issue_key}")

                # Check if this is a creation response
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
                ]

                is_creation = any(
                    keyword in response_lower for keyword in creation_keywords
                )

                # If this is a creation and we have channel_id, post success message and track
                if is_creation and channel_id and issue_key:
                    logger.info(
                        "This is a creation response - posting success message and tracking"
                    )

                    try:
                        from slack_sdk import WebClient

                        client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

                        # Post the success message
                        success_response = client.chat_postMessage(
                            channel=channel_id,
                            text=formatted_response,
                            unfurl_links=True,
                            unfurl_media=True,
                        )

                        if success_response["ok"]:
                            success_timestamp = success_response["ts"]
                            logger.info(
                                f"Posted success message with timestamp: {success_timestamp}"
                            )

                            # Now find the original user trigger by working backwards
                            user_trigger_timestamp = (
                                self.find_user_trigger_from_success(
                                    channel_id, success_timestamp, user_query.query
                                )
                            )

                            if user_trigger_timestamp:
                                logger.info(
                                    f"Found user trigger timestamp: {user_trigger_timestamp}"
                                )

                                # Save tracking data with user trigger timestamp
                                channel_name = self.utils.get_channel_name(channel_id)
                                self.utils.save_slack_tracking_data(
                                    message_id=user_trigger_timestamp,  # Real user message!
                                    channel_id=channel_id,
                                    channel_name=channel_name,
                                    issue_key=issue_key,
                                )
                                logger.info(
                                    f"✅ Successfully saved tracking data for issue {issue_key}"
                                )
                            else:
                                logger.warning(
                                    "Could not find user trigger - using success timestamp as fallback"
                                )
                                channel_name = self.utils.get_channel_name(channel_id)
                                self.utils.save_slack_tracking_data(
                                    message_id=success_timestamp,
                                    channel_id=channel_id,
                                    channel_name=channel_name,
                                    issue_key=issue_key,
                                )

                    except Exception as post_error:
                        logger.error(f"Error posting success message: {post_error}")

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

    # STEP 3: Add method to find user trigger by working backwards from success message
    def find_user_trigger_from_success(
        self, channel_id: str, success_timestamp: str, original_query: str
    ) -> str:
        """Find user trigger by working backwards from success message"""
        try:
            from slack_sdk import WebClient

            client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

            # Get recent message history
            response = client.conversations_history(
                channel=channel_id, limit=30, inclusive=True
            )

            if not response["ok"]:
                logger.error(f"Failed to get message history: {response['error']}")
                return None

            messages = response["messages"]
            messages.sort(key=lambda x: float(x["ts"]))  # Sort by timestamp

            # Find success message index
            success_index = None
            for i, msg in enumerate(messages):
                if msg["ts"] == success_timestamp:
                    success_index = i
                    break

            if success_index is None:
                logger.warning("Could not find success message in history")
                return None

            # Work backwards to find pattern: success → processing → user trigger
            query_words = original_query.lower().split()[:5]

            for i in range(success_index - 1, -1, -1):  # Go backwards
                msg = messages[i]
                text = msg.get("text", "").lower()

                # Skip bot messages
                if msg.get("bot_id") or msg.get("user") in ["bot", None]:
                    continue

                # Look for processing message
                if "processing" in text or "⏳" in text:
                    logger.info(f"Found processing message at index {i}")
                    continue  # Keep looking for user message before processing

                # Look for user message that matches query
                if any(word in text for word in query_words if len(word) > 3):
                    logger.info(f"Found matching user trigger message: {msg['ts']}")
                    return msg["ts"]

                # Don't look too far back
                if success_index - i > 10:
                    break

            logger.warning("Could not find user trigger message")
            return None

        except Exception as e:
            logger.error(f"Error finding user trigger: {e}")
            return None
