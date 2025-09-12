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
            logger.info(f"=== DEBUG START ===")
            logger.info(f"Processing query: {user_query.query}")
            logger.info(f"Message ID (AI response): {message_id}")

            # Just get raw chat history - no interpretation
            chat_history_string = self.utils.extract_chat(channel_id)
            refined_query = await self.refactor_query_with_context(
                user_query.query, chat_history_string
            )
            # Give everything raw to the agent - let IT figure it out
            content = f"""
        USER QUERY: {refined_query}

        CONVERSATION HISTORY: 
        {chat_history_string}

        INSTRUCTIONS:
        - Analyze the user query and conversation history to understand the request
        - Extract any mentioned assignees, issue keys, priorities, etc. from the text
        - If this refers to previous conversations (like "assign it to X"), look at the history to understand context
        - For creation requests without assignees, ask who to assign to
        - Call the appropriate tool based on your analysis
        """

            # Execute with single agent - no preprocessing!
            result = self.jira_agent.invoke(
                {"messages": [{"role": "user", "content": content}]}
            )

            if result and "messages" in result and len(result["messages"]) > 0:
                final_message = result["messages"][-1]
                response_content = final_message.content
                formatted_response = self.utils.format_for_slack(response_content)

                logger.info(f"Formatted response: {formatted_response[:200]}...")

                # Extract issue key from response for tracking
                issue_key = self.utils.extract_issue_key_from_response(
                    formatted_response
                )
                logger.info(f"Extracted issue key: {issue_key}")

                # NEW LOGIC: Find the original slash command message
                original_slash_message_id = None
                if channel_id and message_id:
                    try:
                        original_slash_message_id = self.find_original_slash_command(
                            channel_id, message_id, user_query.query
                        )
                        logger.info(
                            f"Found original slash command message ID: {original_slash_message_id}"
                        )
                    except Exception as e:
                        logger.error(f"Error finding original slash command: {e}")

                # Use original slash command message ID for tracking
                tracking_message_id = original_slash_message_id or message_id

                # DEBUG: Check all conditions
                logger.info(f"=== TRACKING CONDITIONS CHECK ===")
                logger.info(f"channel_id present: {bool(channel_id)}")
                logger.info(f"issue_key present: {bool(issue_key)}")
                logger.info(f"tracking_message_id present: {bool(tracking_message_id)}")
                logger.info(f"formatted_response present: {bool(formatted_response)}")

                # Only save tracking data for CREATION operations - check response for creation keywords
                if (
                    channel_id
                    and issue_key
                    and tracking_message_id
                    and formatted_response
                ):
                    logger.info("All conditions met, checking for creation keywords...")

                    response_lower = formatted_response.lower()
                    logger.info(f"Response (lowercase): {response_lower[:200]}...")

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
                        "created",  # More general keyword
                        "issue has been created",
                        "ticket created",
                    ]

                    # Check if this is a creation response
                    found_keywords = [
                        keyword
                        for keyword in creation_keywords
                        if keyword in response_lower
                    ]
                    logger.info(f"Found creation keywords: {found_keywords}")

                    is_creation = any(
                        keyword in response_lower for keyword in creation_keywords
                    )
                    logger.info(f"Is creation: {is_creation}")

                    # TEMPORARY: Force save for debugging
                    if True:  # Change this to 'if is_creation:' after debugging
                        try:
                            logger.info("Attempting to save tracking data...")
                            channel_name = self.utils.get_channel_name(channel_id)
                            logger.info(f"Channel name: {channel_name}")

                            self.utils.save_slack_tracking_data(
                                message_id=tracking_message_id,  # Use original slash command message
                                channel_id=channel_id,
                                channel_name=channel_name,
                                issue_key=issue_key
                                or "DEBUG-KEY",  # Use debug key if none found
                            )
                            logger.info(
                                f"✅ Successfully saved tracking data for issue {issue_key}"
                            )
                        except Exception as e:
                            logger.error(
                                f"❌ Error saving tracking data: {e}", exc_info=True
                            )
                    else:
                        logger.info(
                            "❌ Not a creation response - no tracking data saved"
                        )
                else:
                    logger.info("❌ Conditions not met for saving tracking data:")
                    if not channel_id:
                        logger.info("  - Missing channel_id")
                    if not issue_key:
                        logger.info("  - Missing issue_key")
                    if not tracking_message_id:
                        logger.info("  - Missing tracking_message_id")
                    if not formatted_response:
                        logger.info("  - Missing formatted_response")

                logger.info(f"=== DEBUG END ===")

                return {
                    "success": True,
                    "message": "Jira operation completed",
                    "data": formatted_response,
                    "query": user_query.query,
                    "issue_key": issue_key,
                    "original_message_id": original_slash_message_id,  # Include for debugging
                }
            else:
                logger.error("❌ No response generated from Jira processing")
                return {
                    "success": False,
                    "message": "No response generated from Jira processing",
                    "data": None,
                    "query": user_query.query,
                }

        except Exception as e:
            logger.error(f"❌ Error processing query: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to process Jira query: {str(e)}",
                "data": None,
                "query": user_query.query,
            }

    def find_original_slash_command(
        self, channel_id: str, current_message_id: str, query: str
    ) -> str:
        """
        Find the original slash command message by looking backwards through message history

        Message chain:
        1. /jiratest create story... (original slash command - WANT THIS)
        2. ⏳ Processing... (bot processing message)
        3. ✅ [answer] (AI response message - current_message_id)
        """
        try:
            from slack_sdk import WebClient
            import os

            client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

            # Get recent message history (last 20 messages should be enough)
            response = client.conversations_history(
                channel=channel_id, limit=20, inclusive=True
            )

            if not response["ok"]:
                logger.error(f"Failed to get message history: {response['error']}")
                return None

            messages = response["messages"]
            logger.info(f"Found {len(messages)} messages in history")

            # Sort messages by timestamp (oldest first)
            messages.sort(key=lambda x: float(x["ts"]))

            # Find the current message index
            current_index = None
            for i, msg in enumerate(messages):
                if msg["ts"] == current_message_id:
                    current_index = i
                    break

            if current_index is None:
                logger.warning(
                    f"Could not find current message {current_message_id} in history"
                )
                # Fallback: look for processing message and slash command pattern
                return self.find_slash_command_fallback(messages, query)

            logger.info(f"Current message found at index {current_index}")

            # Look backwards from current message to find the pattern:
            # 1. Current message (AI response) - index current_index
            # 2. Processing message - index current_index-1
            # 3. Slash command or user message - index current_index-2

            # Check if there's a processing message before current
            if current_index >= 1:
                processing_msg = messages[current_index - 1]
                logger.info(
                    f"Potential processing message: {processing_msg.get('text', '')[:50]}"
                )

                # Check if it looks like a processing message
                processing_text = processing_msg.get("text", "").lower()
                if "processing" in processing_text or "⏳" in processing_text:
                    logger.info("Found processing message")

                    # Look for slash command or user message before processing
                    if current_index >= 2:
                        slash_msg = messages[current_index - 2]
                        logger.info(
                            f"Potential slash command: {slash_msg.get('text', '')[:50]}"
                        )

                        # Check if this looks like the original command
                        slash_text = slash_msg.get("text", "")
                        if slash_text.startswith("/jiratest") or any(
                            word in slash_text.lower()
                            for word in query.lower().split()[:3]
                        ):
                            logger.info(
                                f"Found matching slash command: {slash_msg['ts']}"
                            )
                            return slash_msg["ts"]

            # Fallback: search for any message that looks like the slash command
            return self.find_slash_command_fallback(messages, query)

        except Exception as e:
            logger.error(f"Error finding original slash command: {e}")
            return None

    def find_slash_command_fallback(self, messages, query):
        """Fallback method to find slash command by content matching"""
        try:
            query_words = query.lower().split()[:5]  # First 5 words of query

            for msg in reversed(messages):  # Search from newest to oldest
                text = msg.get("text", "").lower()

                # Look for slash command
                if text.startswith("/jiratest"):
                    logger.info(f"Found slash command by prefix: {msg['ts']}")
                    return msg["ts"]

                # Look for message containing similar words to query
                if len(query_words) >= 2:
                    matches = sum(1 for word in query_words if word in text)
                    if matches >= 2:  # At least 2 matching words
                        logger.info(
                            f"Found potential command by content match: {msg['ts']}"
                        )
                        return msg["ts"]

            logger.warning("Could not find slash command in fallback search")
            return None

        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return None
