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
                self.get_project_epics_sync,
                self.get_slack_reporter_for_ticket_sync,  # ← ADD THIS BACK
            ],
            prompt=self._get_unified_prompt(),
        )

    def _get_unified_prompt(self) -> str:
        """Enhanced unified prompt for intelligent content extraction and Jira operations."""
        return """
    You are a Jira assistant that helps people create, update, and manage tickets through natural conversation.

    🚨 CRITICAL CONVERSATION STATE TRACKING 🚨

    You MUST maintain conversation state across messages. If you asked the user a question, their next response is answering that question.

    **STATE 1: You just asked "Who should work on this?" and showed a user list**
    → User's next message = THE ASSIGNEE NAME
    → Immediately create the ticket with that name as assignee
    → DO NOT ask any follow-up questions about who they're referring to

    **STATE 2: You just created a ticket and waiting for confirmation**
    → User's next message might be asking to assign it
    → Update that ticket immediately

    **Common Patterns When User Responds After Your Question:**
    - "fahad" → assignee="Fahad"
    - "john" → assignee="john" 
    - "assign to sarah" → assignee="sarah"
    - Just a name = They're giving you the assignee you asked for

    **NEVER respond with "It seems like you're referring to..." or "Could you provide more details" when you JUST asked them for an assignee and they gave you a name!**

    Your Main Jobs

    1. Create tickets - Turn user requests into proper Jira tickets with good titles and descriptions
    2. Update tickets - Change existing tickets when users provide the ticket ID (like AI-123)
    3. Handle follow-up questions - Understand when users refer back to previous tickets

    AUTOMATIC REPORTER ASSIGNMENT (SILENT)

    When creating tickets, silently match the Slack user to a Jira reporter:
    1. Call get_slack_reporter_for_ticket_sync(slack_username="<name from context>")
    2. Match the Slack username to best Jira user
    3. Use their account_id as reporter_email in create_issue_sync
    4. NEVER mention this to the user

    Example (Internal):
    Context: SLACK USER: Fahad Ahmed
    You: Call get_slack_reporter_for_ticket_sync(slack_username="Fahad Ahmed")
    Response: Found "Fahad" with account_id "557058:abc123"
    You: Call create_issue_sync(..., reporter_email="557058:abc123")

    TICKET CREATION WORKFLOW - FOLLOW THIS EXACTLY

    **STEP 1: User asks to create ticket**
    → Extract: summary, description, issue type
    → Check: Do they mention an assignee?

    **STEP 2A: If NO assignee mentioned**
    → Call get_project_assignable_users_sync
    → Show user list
    → Ask: "Who should work on this? Here are the available people: [list]"
    → REMEMBER: You are now in WAITING FOR ASSIGNEE state

    **STEP 2B: If assignee mentioned**
    → Call create_issue_sync immediately with all details

    **STEP 3: User responds with a name**
    → This is the assignee!
    → Check if name exists in the list you showed
    → Call create_issue_sync immediately
    → DO NOT ask "what do you mean?" or "who is this for?"

    **Example Conversation Flow:**

    User: "create ticket on mobile stole"
    You: Call get_project_assignable_users_sync → "Who should work on this? Available people: Hamza, Waqas, Fahad..."

    User: "fahad"
    You: [RECOGNIZE: They just gave me the assignee I asked for!]
        Call create_issue_sync(
            assignee_email="Fahad",
            summary="Mobile theft incident",
            description_text="Report of mobile device theft. Investigation needed to document incident details and determine next steps.",
            issue_type_name="Story"
        )
        Response: "I've created the ticket: AI-XXX - Mobile theft incident (assigned to Fahad)"

    ❌ WRONG: "It seems like you're referring to Fahad..." 
    ✅ RIGHT: Just create the ticket!

    ADVANCED CHAT HISTORY ANALYSIS

    MULTI-TICKET DETECTION:
    - If conversation history contains multiple distinct issues, create separate tickets
    - Look for user mentions and auto-assign if they exist in Jira
    - Match discussed issues with mentioned users

    DUPLICATE DETECTION:
    Check for duplicates before creating tickets using:
    - LEVEL 1: Exact issue key mentions (AI-3340, etc.)
    - LEVEL 2: Semantic similarity (payment/stripe/billing keywords)
    - LEVEL 3: Context (issues discussed in last 60 minutes)

    Creating New Tickets

    ISSUE TYPE DEFAULT RULE:
    Always create tickets as "Story" unless user explicitly mentions "bug"

    CRITICAL: Always generate proper descriptions:
    - Bugs: "User reported [issue]. [Impact]. Investigation needed..."
    - Stories: "[Feature requested]. [Purpose]. Implementation needed..."
    - Tasks: "[Work requested]. [Context]. Action needed..."

    NEVER leave description empty!

    Understanding Context

    Pay attention to conversation flow:
    - "assign it to sarah" = assign the ticket we just talked about
    - "update that ticket" = update the most recent ticket
    - After showing user list, expect their response to be picking someone

    When to Use Each Tool

    create_issue_sync - Create new ticket
    - ALWAYS provide: assignee, summary, description_text, issue_type
    - Optional: story_points, epic_key
    - Reporter is automatic - don't include reporter_email

    update_issue_sync - Update existing ticket
    - Need: ticket ID (AI-123)
    - Optional: any field to update

    get_project_assignable_users_sync - Show available assignees
    - Use when asking for assignees

    delete_issue_sync - Delete ticket
    - Need: ticket ID

    Response Style

    Be conversational:
    - "I've created the story for you: AI-456"
    - "I've assigned it to Sarah"

    Don't be robotic:
    - "Ticket creation successful" ✗
    - "Operation completed" ✗

    Important Rules

    1. **MAINTAIN STATE** - Remember what you just asked the user
    2. If you showed user list and they respond with a name, create the ticket immediately
    3. Never create tickets without assignees
    4. Always generate meaningful descriptions
    5. Use exact names users give you
    6. Check chat history for duplicates
    7. NEVER ask about reporter

    Your goal is to make Jira operations natural while ensuring tickets are properly created with meaningful content.
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
        story_points: int = None,
        epic_key: str = None,
        force_update_description_after_create: bool = True,
    ) -> dict:
        """
        Create a new Jira ticket/issue.

        🔴 ASSIGNEE IS MANDATORY 🔴
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
            story_points: Optional story points (1, 2, 3, 5, 8, 13, etc.)
            epic_key: Optional epic key to link this ticket to (e.g., "AI-123")

        Returns:
            dict: Success with ticket details OR error if assignee missing

        Examples:
            ✅ create_issue_sync(assignee_email="john", summary="Fix login bug", story_points=5)
            ✅ create_issue_sync(assignee_email="sarah", summary="API work", epic_key="AI-100")
            ❌ create_issue_sync(summary="Fix bug") # Will ask for assignee
        """
        # CRITICAL FIX: Get the slack username from the instance variable
        return self.utils.create_issue_implementation(
            project_name_or_key,
            summary,
            description_text,
            assignee_email,
            priority_name,
            reporter_email,  # Agent will pass account_id here
            issue_type_name,
            sprint_name,
            story_points,
            epic_key,
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
        story_points: int = None,  # NEW: Optional story points
        epic_key: str = None,  # NEW: Optional epic to link to
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
            story_points,  # NEW
            epic_key,  # NEW
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

    def get_project_epics_sync(self, project_key: str = "AI") -> dict:
        """
        Get list of epics for a project to allow users to link tickets to epics.
        Now uses the jira Python library for reliable epic retrieval.

        Args:
            project_key: Project key like "AI", "BUN", etc. (defaults to "AI")

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

    def get_slack_reporter_for_ticket_sync(
        self, slack_username: str, project_key: str = "AI"
    ) -> dict:
        """
        Get Jira users list to match Slack username for reporter assignment.
        """
        try:
            users = self.utils.get_project_users(project_key, max_results=100)
            print(
                f"🔧 TOOL CALLED: get_slack_reporter_for_ticket_sync with slack_username='{slack_username}'"
            )
            print(users)
            if not users:
                return {
                    "success": False,
                    "slack_username": slack_username,
                    "jira_users": [],
                    "message": f"No Jira users found in project {project_key}",
                }

            user_list = []
            for user in users:
                user_list.append(
                    {
                        "display_name": user["displayName"],
                        "email": user.get("emailAddress", ""),
                        "account_id": user["accountId"],
                    }
                )

            return {
                "success": True,
                "slack_username": slack_username,
                "jira_users": user_list,
                "message": f"Match '{slack_username}' to the best user and use their account_id for reporter_email in create_issue_sync.",
            }
        except Exception as e:
            logger.error(f"Error in get_slack_reporter_for_ticket_sync: {e}")
            return {
                "success": False,
                "slack_username": slack_username,
                "jira_users": [],
                "message": f"Error: {str(e)}",
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
            self.current_slack_username = slack_username
            print("User Name: ", slack_username)
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

        SLACK USER: {slack_username if slack_username else "Unknown"}

        CONVERSATION HISTORY: 
        {chat_history_string}

        INSTRUCTIONS:
        - The SLACK USER is who created this request
        - For ticket creation, call get_slack_reporter_for_ticket_sync silently to match them to Jira
        - Use the refined query as your primary instruction
        - Extract any mentioned assignees, issue keys, priorities, etc.
        - Call the appropriate tool based on your analysis
        """

            # Execute with single agent using refined query
            result = self.jira_agent.invoke(
                {"messages": [{"role": "user", "content": content}]}
            )

            self.current_slack_username = None

            if result and "messages" in result and len(result["messages"]) > 0:
                final_message = result["messages"][-1]
                response_content = final_message.content
                formatted_response = self.utils.format_for_slack(response_content)

                # Extract issue key from response
                issue_key = self.utils.extract_issue_key_from_response(
                    formatted_response
                )
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
                    "created the ticket",  # Added for multi-step
                    "ticket for you:",  # Added for multi-step
                    "assigned to",  # Added for multi-step (when assignment happens)
                    "created the tickets",
                ]

                is_creation = any(
                    keyword in response_lower for keyword in creation_keywords
                )
                logger.info(f"Is creation: {is_creation}")
                logger.info(f"Response check: {response_lower[:100]}...")

                # TRACKING LOGIC - handle both direct and multi-step scenarios
                if is_creation and channel_id and issue_key:
                    logger.info("This is a creation response - handling tracking")

                    try:
                        # Determine if this is from slash command or regular message
                        is_slash_command = message_id == "SLASH_COMMAND"

                        if is_slash_command:
                            # For slash commands, find the user trigger by working backwards from recent messages
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

    # STEP 3: Add method to find recent user trigger (for multi-step conversations)
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
