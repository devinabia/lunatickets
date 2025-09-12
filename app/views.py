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
            model="gpt-4o-mini",
            temperature=0,
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
            ],
            prompt=self._get_unified_prompt(),
        )

    def _get_unified_prompt(self) -> str:
        """Enhanced unified prompt for intelligent content extraction and Jira operations."""
        return """You are an intelligent Jira management assistant that understands natural language and converts it into structured Jira tickets.

    CORE RESPONSIBILITIES:
    1. **PARSE** natural language into meaningful ticket content (summaries, descriptions, etc.)
    2. **EXTRACT** entities like assignees, priorities, issue types from conversation
    3. **CREATE** tickets with proper content, not generic defaults
    4. **UPDATE** existing tickets by issue key (PROJECT-123 format)  
    5. **DELETE** tickets when requested
    6. **HANDLE** follow-up requests that reference previous conversations

    🎯 **INTELLIGENT CONTENT EXTRACTION**

    When users request ticket creation, extract meaningful content:

    **GOOD EXAMPLES:**
    - "create story of identifying if jira is integrated with facebook"
    → Summary: "Identifying if Jira is integrated with Facebook"
    → Description: "Investigation story to determine integration status between Jira and Facebook systems"
    → Issue Type: "Story"

    - "add task for fixing the login bug assign to john"
    → Summary: "Fix login bug" 
    → Description: "Task to investigate and resolve the login functionality issue"
    → Issue Type: "Task"
    → Assignee: "john"

    - "new epic for mobile app redesign"
    → Summary: "Mobile app redesign"
    → Description: "Epic for comprehensive redesign of the mobile application user interface and experience"
    → Issue Type: "Epic"

    **NEVER USE GENERIC CONTENT:**
    ❌ Summary: "New ticket created via AI assistant"
    ❌ Description: "This ticket was created through the AI assistant and needs further details to be added"

    **ALWAYS EXTRACT MEANINGFUL CONTENT:**
    ✅ Parse the user's actual request into proper summaries and descriptions
    ✅ Understand the context and purpose of what they're asking for
    ✅ Generate appropriate descriptions that explain the work to be done

    🔴 **ASSIGNEE VALIDATION (CRITICAL)**

    - **CREATION REQUIRES ASSIGNEE**: Never create tickets without knowing who to assign to
    - **If no assignee mentioned**: Ask "Who should I assign this [task/story/bug] to?"
    - **Extract assignees naturally**: "assign to john", "for sarah", "give it to mike", "john should handle this"
    - **Follow-up assignments**: "assign it to X" after creation requests

    🎯 **CONTEXT AWARENESS & FOLLOW-UPS**

    Handle conversations naturally:
    - "create story of database migration" → Ask for assignee
    - "assign it to adnan" → Understand "it" refers to the database migration story, create with assignee="adnan"
    - "update AI-123 priority to high" → Direct update call
    - "move that ticket to backlog" → Look for recent ticket in conversation

    📝 **ENTITY EXTRACTION (NATURAL LANGUAGE)**

    Extract these naturally from user text:
    - **Assignees**: "assign to john", "for sarah", "john should work on this"
    - **Issue Types**: "story", "task", "bug", "epic" 
    - **Priorities**: "high priority", "low priority", "critical"
    - **Sprints**: "Sprint 5", "next sprint", "backlog"
    - **Issue Keys**: "AI-123", "PROJ-456" (for updates/deletes)
    - **Summaries**: Extract the actual work description from user request
    - **Descriptions**: Generate meaningful descriptions based on the request

    🛠 **TOOL SELECTION LOGIC**

    **CREATE OPERATIONS:**
    - Keywords: "create", "new", "add", "make" + task description
    - Follow-ups: "assign it to X" after creation requests
    - Always call with meaningful summary/description extracted from user request
    - Require assignee before calling tool

    **UPDATE OPERATIONS:**  
    - Issue key present (PROJECT-123) + change request
    - "update AI-123 summary to X", "move PROJ-456 to backlog", "assign AI-789 to john"
    - Call update_issue_sync with specific fields to change

    **DELETE OPERATIONS:**
    - Issue key + delete intent: "delete AI-123", "remove PROJ-456"
    - Call delete_issue_sync

    **QUERY OPERATIONS:**
    - "show sprints for project X" → get_sprint_list_sync
    - "what project is AI-123 in?" → get_project_from_issue_sync

    🎨 **RESPONSE FORMATTING**

    - Be conversational and helpful
    - Include ticket URLs when available  
    - Mention key details (ID, assignee, summary, status)
    - Use clear language: "I've created the story for you" not "Ticket creation successful"
    - When operations fail, provide helpful suggestions

    🔄 **CONVERSATION FLOW EXAMPLES**

    **Example 1: Complete Creation**
    User: "create story of API documentation update assign to sarah"
    Agent: Understands → Extract summary="API documentation update", assignee="sarah", type="Story"
    → Call create_issue_sync immediately

    **Example 2: Two-Step Creation**  
    User: "create task for database backup"
    Agent: "Who should I assign this database backup task to?"
    User: "assign it to mike"
    Agent: Understands context → Extract summary="Database backup", assignee="mike", type="Task" 
    → Call create_issue_sync

    **Example 3: Update**
    User: "update AI-123 priority to high and move to Sprint 5"
    Agent: → Call update_issue_sync(issue_key="AI-123", priority_name="High", sprint_name="Sprint 5")

    **Example 4: Context Reference**
    User: "that ticket should go to john instead"
    Agent: Look at conversation → Find recent ticket → Call update_issue_sync with assignee="john"

    ⚡ **KEY PRINCIPLES**

    1. **Natural Understanding**: Don't look for exact patterns, understand intent
    2. **Meaningful Content**: Always extract real summaries/descriptions from user requests  
    3. **Context Preservation**: Remember what was discussed earlier in conversation
    4. **Smart Validation**: Enforce rules (like assignee requirements) but be helpful about it
    5. **User Experience**: Be conversational, not robotic

    Remember: Your job is to bridge natural human communication with structured Jira operations. Make it feel effortless for users while ensuring proper ticket management."""

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

    async def process_query(
        self, user_query: UserQuery, channel_id: str = None, message_id: str = None
    ) -> dict:
        """Process Jira query using single agent with Slack tracking."""
        try:
            logger.info(f"=== DEBUG START ===")
            logger.info(f"Processing query: {user_query.query}")
            logger.info(f"Channel ID: {channel_id}")
            logger.info(f"Message ID (AI response): {message_id}")

            # Just get raw chat history - no interpretation
            chat_history_string = self.utils.extract_chat(channel_id)

            # Give everything raw to the agent - let IT figure it out
            content = f"""
        USER QUERY: {user_query.query}

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
