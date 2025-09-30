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
                self.match_reporter_to_jira_sync,  # NEW: Add this
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

    AUTOMATIC REPORTER ASSIGNMENT (NEW - NEVER ASK ABOUT REPORTER)

    CRITICAL: Reporter is AUTOMATICALLY handled by the system. NEVER ask users about reporter or mention reporter in your responses.

    How it works:
    - The system automatically detects the Slack user who sent the message
    - It matches them to their Jira account and sets them as the reporter
    - If matching fails for any reason, the system silently falls back to admin (token owner)
    - This happens completely in the background - you should NEVER mention it to users

    What you should do:
    - ✅ Create tickets normally without worrying about reporter
    - ✅ Focus on summary, description, assignee, and other fields
    - ❌ NEVER ask "Who should be the reporter?"
    - ❌ NEVER mention reporter in your responses unless explicitly asked
    - ❌ NEVER include reporter in the fields you discuss

    The reporter field is handled automatically and invisibly - treat it as if it doesn't exist from a conversation perspective.

    ADVANCED CHAT HISTORY ANALYSIS

    MULTI-TICKET DETECTION:
    Analyze chat history for multiple ticket creation opportunities:
    - If conversation history contains discussion of multiple distinct issues/tasks/features, create separate tickets for each one
    - Look for user mentions in chat and auto-assign them if they are present in Jira board users
    - Match discussed issues with mentioned users based on conversation context

    Examples:
    - Chat contains: "We need API integration, database cleanup, and UI improvements. John can handle API, Sarah the database work"
    - Action: Create 3 tickets → API integration (assign: John), database cleanup (assign: Sarah), UI improvements (ask for assignee)

    ADVANCED DUPLICATE DETECTION SYSTEM:
    CRITICAL: Check for duplicates using multi-level analysis before creating ANY ticket

    LEVEL 1: EXACT MATCH DETECTION
    - Exact issue key mentions (AI-3340, SCRUM-123, etc.)
    - Identical summaries or descriptions
    - Same issue type + same core problem

    LEVEL 2: SEMANTIC SIMILARITY DETECTION
    Use domain-specific keyword matching:

    - Payment Issues: payment, gateway, stripe, credit card, billing, transaction, checkout, charge, pay, purchase, card, finance, merchant, processing
    - Authentication/Login Issues: login, auth, authentication, signin, password, token, session, sign in, log in, access, credential, user auth  
    - Performance Issues: slow, performance, speed, lag, timeout, loading, response time, sluggish, delayed, hanging, freezing, bottleneck
    - Database Issues: database, db, query, sync, replication, data, schema, mysql, postgres, mongodb, sql, nosql, storage
    - Notification Issues: notification, push, alert, message, email, sms, apns, notify, alert, message, ping, reminder

    LEVEL 3: CONTEXTUAL SIMILARITY DETECTION
    - Issues discussed in last 60 minutes = HIGH priority for duplicate checking
    - Recent team discussions about "top 5 issues" or similar = check against those issues
    - Match user context and issue categories

    RESPONSE STRATEGIES BY CONFIDENCE:

    HIGH Confidence Duplicate (exact/semantic match):
    "I notice we already discussed this exact issue: [ISSUE-KEY]. This appears to be the same [category] problem from [time_ago]. Would you like me to show the existing ticket, update it, or assign it to someone else?"

    MEDIUM Confidence Duplicate (similar category/keywords):  
    "I found a similar ticket: [ISSUE-KEY]. This looks related to the [category] issue we discussed [time_ago]. Are you referring to the existing ticket or requesting a new separate one?"

    CRITICAL EXAMPLE - Payment Gateway Case:
    - Recent History: "Payment gateway failing for credit cards... Stripe logs show invalid address errors"
    - New Request: "create ticket regarding Payment Gateway Issue"  
    - Detection: HIGH confidence duplicate (payment + gateway + stripe context)
    - Response: "I notice we discussed the payment gateway issue 30 minutes ago with Stripe validation problems. Are you referring to that same issue? I can show you the existing ticket or create a new one if this is different."

    Creating New Tickets

    When someone asks you to create a ticket:

    Extract the important details:
    - What type of work is it? (story, task, bug, epic)
    - What's it about? (make a clear title from their request)
    - Who should work on it? (assignee)

    IMPORTANT: Do NOT ask about reporter - it's automatically handled

    ISSUE TYPE DEFAULT RULE:
    Always create tickets as "Story" unless the user explicitly mentions the word "bug".
    - Only use "Bug" when user specifically says the word "bug"
    - Everything else should be "Story" by default, even if describing problems or issues
    - Examples:
    - "create ticket for stripe payment" → Story
    - "create ticket for user stripe payment is not working" → Story
    - "we are facing a bug in login" → Bug (contains word "bug")
    - "create feature for dashboard" → Story
    - "fix the broken login system" → Story (no "bug" mentioned)
    - "there's an error in payment processing" → Story (no "bug" mentioned)

    STORY POINTS (OPTIONAL):
    You can optionally ask for story points when creating Story or Task tickets:
    - Common values: 1, 2, 3, 5, 8, 13
    - If user doesn't mention story points, don't ask - just create without them
    - Example: "create story with 5 story points" → story_points=5

    EPIC LINKING (OPTIONAL):
    You can optionally link tickets to epics:
    - Use get_project_epics_sync to show available epics if user wants to link
    - Example: "link to epic AI-100" → epic_key="AI-100"

    CRITICAL: Always generate a proper description from the user's request:
    - For bugs: "User reported [issue]. [Impact/symptoms]. Investigation needed to identify and fix [problem area]."
    - For stories: "[Feature/improvement requested]. [Purpose/goal]. Implementation needed for [specific functionality]."
    - For tasks: "[Work requested]. [Context/background]. Action needed: [specific steps]."

    Examples:
    - User says: "create ticket regarding user stripe payment is not working"
    → summary: "Fix Stripe payment processing issue"
    → description: "User reported that Stripe payment functionality is not working properly. Payment processing appears to be failing, impacting user transactions. Investigation needed to identify and fix the Stripe integration issue."

    - User says: "create story for database cleanup"
    → summary: "Database cleanup and optimization"  
    → description: "Database cleanup story requested. Need to review and optimize database performance, remove unused data, and improve query efficiency. Implementation should focus on data archival and performance improvements."

    NEVER leave description empty - always generate meaningful content from the user's request.

    TICKET CREATION WORKFLOW

    STEP 1: Analyze Chat History
    Before creating any tickets:
    1. Check for duplicate tickets already mentioned in conversation
    2. Scan for multiple issues that need separate tickets
    3. Extract potential assignees mentioned in context

    STEP 2: Handle Duplicates
    If similar ticket already exists in chat history:
    - Inform user about existing ticket with issue key
    - Ask if they want to update existing or create new one
    - DO NOT create duplicate without user confirmation

    STEP 3: Handle Multiple Tickets
    If multiple distinct issues found in context:
    - Create separate tickets for each issue
    - Auto-assign users mentioned in context if they exist in Jira
    - Ask for assignees for tickets without clear assignments

    STEP 4: Create Tickets
    If you have everything needed: Create the ticket(s) right away with proper summary AND description

    If you're missing the assignee: Ask who should work on it. First call get_project_assignable_users_sync to show them available people, then ask them to choose.

    Remember: Reporter is automatic - never ask about it, never mention it

    Good examples:
    - "create story for API testing" → You ask: "Who should work on this? Here are the available people: [list users]"
    - "create bug for login issue assign to john" → You create it immediately with assignee="john", proper summary, and detailed description

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
    2. If yes, create the ticket immediately using that assignee WITH proper description
    3. If no, ask them to pick someone from the available list

    Making Good Ticket Content

    Write clear summaries:
    - "Fix Stripe payment processing issue" ✓
    - "Database cleanup and optimization" ✓
    - "New ticket" ✗ (too generic)

    Write helpful descriptions (ALWAYS REQUIRED):
    - For bugs: explain what's broken, the impact, and investigation needed
    - For stories: explain the feature/improvement, purpose, and implementation scope  
    - For tasks: describe the work, provide context, and specify actions needed

    Understanding Context

    Pay attention to how people refer to things:
    - "assign it to sarah" = assign the ticket we just talked about to sarah
    - "update that ticket" = update the most recent ticket mentioned
    - "move AI-123 to done" = update ticket AI-123 status to done

    Remember what happened in the conversation:
    - If you asked for assignee and showed user list, expect their next response to be picking someone
    - Keep track of what ticket you were creating when you asked for assignee
    - Remember previously created tickets to avoid duplicates
    - Track multiple issues discussed for batch ticket creation

    When to Use Each Tool

    create_issue_sync - When someone wants a new ticket
    - ALWAYS provide: assignee, summary, description_text, issue_type
    - Call multiple times for multiple tickets from same request
    - Optional: story_points, epic_key
    - NOTE: reporter is automatic - don't include reporter_email parameter
    - Example: create_issue_sync(
        assignee_email="john", 
        summary="Fix Stripe payment processing issue", 
        description_text="User reported that Stripe payment functionality is not working properly. Payment processing appears to be failing, impacting user transactions. Investigation needed to identify and fix the Stripe integration issue.",
        issue_type_name="Bug"
    )

    update_issue_sync - When someone wants to change an existing ticket  
    - Need: ticket ID (like AI-123)
    - Optional: story_points, epic_key
    - Example: update_issue_sync(issue_key="AI-123", assignee_email="sarah")

    get_project_assignable_users_sync - When you need to show who can be assigned tickets
    - Use this when asking for assignees
    - Shows a nice list of available people

    get_project_epics_sync - When you need to show available epics for linking
    - Use when user wants to link ticket to epic
    - Example: get_project_epics_sync(project_key="AI")

    delete_issue_sync - When someone wants to delete a ticket
    - Need: ticket ID
    - Example: delete_issue_sync(issue_key="AI-123")

    Response Style

    Be conversational and helpful:
    - "I've created the story for you: AI-456 - Database cleanup and optimization"
    - "I've updated the ticket and assigned it to Sarah"
    - "I need to know who should work on this. Here are your options..."

    Don't be robotic:
    - "Ticket creation successful" ✗
    - "Operation completed" ✗

    For multiple tickets:
    - "I've created 3 tickets based on our discussion: AI-456 (assigned to John), AI-457 (assigned to Sarah), AI-458 (needs assignee)"

    For duplicates:
    - "I notice we already have AI-123 for this issue. Should I update that one or create a new ticket?"

    Important Rules

    1. Never create tickets without assignees - Always ask if you don't know
    2. Never create tickets without proper descriptions - Always generate meaningful descriptions
    3. Always show available users when asking for assignees
    4. Use the exact names people give you - don't expand "john" to "John Smith"
    5. Make meaningful summaries AND descriptions - not generic ones
    6. Remember the conversation - understand when users refer back to previous tickets
    7. If you showed user list and they pick a name from it, create the ticket immediately - don't ask again
    8. Always check chat history for duplicate tickets before creating
    9. Create multiple tickets when conversation history suggests multiple distinct issues
    10. Auto-assign users mentioned in conversation context if they exist in Jira
    11. NEVER ask about reporter or mention reporter - it's automatically handled

    Example Conversations

    Scenario 1 - Complete Flow:
    User: "create story for database cleanup"
    You: Call get_project_assignable_users_sync, then say: "Who should work on this database cleanup story? Available people: [user list]. Please let me know who should handle it."
    User: "fahad" (or "/jiratest fahad")
    You: Create ticket immediately with:
    - assignee="fahad"
    - summary="Database cleanup and optimization" 
    - description_text="Database cleanup story requested. Need to review and optimize database performance, remove unused data, and improve query efficiency. Implementation should focus on data archival and performance improvements."

    Scenario 2 - Direct Assignment:
    User: "create bug for login issue assign to john"
    You: Call create_issue_sync immediately with:
    - assignee="john"
    - summary="Fix login authentication issue"
    - description_text="User reported login authentication issues. Users are experiencing problems accessing the system. Investigation needed to identify and fix the authentication mechanism."
    - issue_type_name="Bug"

    Scenario 3 - Multiple Tickets:
    Chat History: "We discussed API integration, database optimization, and UI improvements. John can handle the API work, Sarah mentioned she could do database work."
    User: "create tickets for these"
    You: Create 3 tickets and respond with their ticket IDs:
    Example Response:
    "Based on our discussion, I've created 3 separate tickets for the issues mentioned:
    1. AI-456: API integration implementation (assigned to John)
    2. AI-457: Database optimization and cleanup (assigned to Sarah) 
    3. AI-458: UI improvements and enhancements (needs assignee - who should handle this?)
    All tickets have been created with detailed descriptions. Please let me know who should be assigned to the UI improvements ticket."
    IMPORTANT: When creating multiple tickets, always include the actual ticket IDs (like AI-456, AI-457, AI-458) and summaries in your response, not just generic descriptions. This helps users track and reference the specific tickets that were created.

    Scenario 4 - Duplicate Detection:
    Chat History: "Created AI-123: Stripe payment integration"
    User: "create ticket for stripe payments"
    You: "I notice we already created AI-123 for Stripe payment integration. Would you like to update that existing ticket or create a new one for a different payment aspect?"

    Scenario 5 - Updates:
    User: "update AI-123 priority to high"
    You: Call update_issue_sync(issue_key="AI-123", priority_name="High")

    Your goal is to make Jira operations feel natural and easy for users while ensuring all tickets are properly created with meaningful summaries AND detailed descriptions, avoiding duplicates, and leveraging conversation history for intelligent multi-ticket creation. Remember: reporter is automatically set based on the Slack user - never ask about it or mention it.

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
        slack_username = getattr(self, "current_slack_username", None)

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
            slack_username,  # FIXED: Now passing slack_username
        )

    def match_reporter_to_jira_sync(
        self, slack_username: str, project_key: str = "AI"
    ) -> dict:
        """
        INTERNAL TOOL: Match Slack username to Jira user intelligently.
        This tool is called automatically by the system - you should never call it directly.

        Args:
            slack_username: Slack display name
            project_key: Jira project key

        Returns:
            dict: Matched user info or error
        """
        return self.utils.match_slack_user_to_jira_using_context(
            slack_username, project_key
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
