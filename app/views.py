import os
import time
import requests
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
import logging
from .schemas import UserQuery
from .utilities.utils import Utils
from .utilities.prompt import Prompt
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class JiraService:
    """Service class for managing Jira tickets using agentic AI with supervisor."""

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
        self.default_project = "LUNA_TICKETS"

        # Session for Jira API calls
        self.session = requests.Session()
        self.session.auth = (self.email, self.token)
        self.session.headers.update(
            {"Accept": "application/json", "Content-Type": "application/json"}
        )

        # Initialize utilities
        self.utils = Utils(self.base_url, self.email, self.token, self.session)

        # Initialize agents and supervisor
        self._setup_agents()

    def _setup_agents(self):
        """Setup all agents and create supervisor."""

        # Create specialized agents
        self.create_agent = create_react_agent(
            model=self.model,
            tools=[self.create_issue_sync],
            name="jira_create_expert",
            prompt=Prompt.TICKET_CREATION_PROMPT,
        )

        self.update_agent = create_react_agent(
            model=self.model,
            tools=[self.update_issue_sync],
            name="jira_update_expert",
            prompt=Prompt.TICKET_UPDATE_PROMPT,
        )

        self.delete_agent = create_react_agent(
            model=self.model,
            tools=[self.delete_issue_sync],
            name="jira_delete_expert",
            prompt=Prompt.TICKET_DELETE_PROMPT,
        )

        # Create supervisor to route between agents
        self.jira_supervisor = create_supervisor(
            [
                self.create_agent,
                self.update_agent,
                self.delete_agent,
            ],
            model=self.model,
            prompt=Prompt.SUPERVISOR_PROMPT,
        )

        # Compile the supervisor workflow
        self.app = self.jira_supervisor.compile()

    def create_issue_sync(
        self,
        project_name_or_key: str,
        summary: str,
        description_text: str,
        assignee_email: str = "",
        priority_name: str = None,
        reporter_email: str = None,
        issue_type_name: str = None,
        force_update_description_after_create: bool = True,
    ) -> dict:
        """
        Create a new Jira ticket/issue. Use this tool ONLY after confirming assignee with the user.

        CRITICAL: DO NOT call this function unless the user has specified who to assign the ticket to.
        If no assignee mentioned in the user query, ask for assignee first before calling this tool.

        Args:
            project_name_or_key: Jira project key (REQUIRED - extract from user query/chat history)
                Examples: "SCRUM", "LUNA_TICKETS", "DevOps", "AI", etc.
            summary: Clear, concise title for the ticket (required)
            description_text: Detailed description of the issue/task (required)
            assignee_email: CONFIRMED assignee from user (REQUIRED - never empty)
                Examples: "john", "john@company.com", "John Smith", "me", "unassigned"
            priority_name: "High" for urgent, "Medium" for normal, "Low" for minor
            reporter_email: Email of person reporting (leave empty "" for current user)
            issue_type_name: "Bug"/"Task"/"Story"/"Epic" based on request type
            force_update_description_after_create: Keep as True

        Returns:
            dict: Comprehensive ticket information with all actual data

        WORKFLOW BEFORE CALLING:
        1. Check if assignee mentioned in user query
        2. If NO assignee → Ask user for assignee first, DO NOT call this function
        3. If assignee mentioned → Call this function immediately

        Examples requiring assignee confirmation first:
        - "create ticket in LUNA_TICKETS" → ASK FOR ASSIGNEE
        - "add bug to DevOps project" → ASK FOR ASSIGNEE
        - "make task for the team" → ASK FOR ASSIGNEE

        Examples ready to create immediately:
        - "create ticket assign to john in SCRUM" → CALL FUNCTION
        - "add bug for sarah@company.com" → CALL FUNCTION
        - "make task assign to me" → CALL FUNCTION
        - "create unassigned ticket in DevOps" → CALL FUNCTION

        Always ask for assignee confirmation before using this tool!
        always return issue key in response
        """

        if issue_type_name is None:
            issue_type_name = self.default_issue_type

        # Use default project if not provided or empty
        if not project_name_or_key or project_name_or_key.strip() == "":
            return {
                "success": False,
                "error": "Project required",
                "message": "Please specify which project to create this ticket in (e.g., SCRUM, DevOps, AI)",
            }

        try:
            project_key = self.utils.resolve_project_key(project_name_or_key)
            normalized_issue_type = self.utils.normalize_issue_type(
                project_key, issue_type_name
            )
            logger.info(
                f"Original issue type: '{issue_type_name}' -> Normalized: '{normalized_issue_type}'"
            )

            allowed = self.utils.get_create_fields(project_key, normalized_issue_type)
            board_info = self.utils.get_board_info(project_key)

            fields = {
                "project": {"key": project_key},
                "summary": summary.strip().strip("'\""),
                "issuetype": {"name": normalized_issue_type},
            }

            # Set description
            if "description" in allowed and description_text:
                fields["description"] = self.utils.text_to_adf(description_text)

            # Smart assignee handling
            assignment_info = {
                "assigned": False,
                "assignee_name": "Unassigned",
                "suggestions": None,
            }

            if "assignee" in allowed and assignee_email and assignee_email.strip():
                assignment_result = self.utils.smart_assign_user(
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
                    # If accountId is None, leave unassigned (which is success)
                else:
                    # Assignment failed, but continue with ticket creation
                    logger.warning(
                        f"Could not assign to '{assignee_email}': {assignment_result['message']}"
                    )
                    assignment_info["suggestions"] = assignment_result.get(
                        "suggestions", ""
                    )

            # Set priority
            if priority_name and "priority" in allowed:
                try:
                    pr_id = self.utils.get_priority_id_by_name(priority_name)
                    fields["priority"] = {"id": pr_id}
                except Exception as e:
                    logger.warning(f"Could not set priority '{priority_name}': {e}")

            # Set reporter
            if reporter_email and "reporter" in allowed:
                try:
                    reporter_id = self.utils.get_account_id(reporter_email)
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
                    self.utils.update_description(issue_key, description_text)
                    logger.info(f"Description updated for {issue_key}")
                except Exception as e:
                    logger.warning(f"Description update failed: {e}")

            # Get final issue details
            final = self.utils.get_issue(issue_key)

            # Return comprehensive information
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
            }

            # Add assignment information if there were issues
            if assignment_info["suggestions"]:
                result["assignment_failed"] = True
                result["user_suggestions"] = assignment_info["suggestions"]
                result["assignment_message"] = (
                    f"Ticket created but could not assign to '{assignee_email}'"
                )

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

    def update_issue_sync(
        self,
        issue_key: str,
        summary: str = None,
        description_text: str = None,
        assignee_email: str = None,
        priority_name: str = None,
        due_date: str = None,
        start_date: str = None,  # Keep parameter for compatibility but ignore it
        issue_type_name: str = None,
        labels: str = None,  # Comma-separated string
    ) -> dict:
        """
        Update an existing Jira ticket/issue. Use this tool when the user wants to:
        - Modify any field of an existing ticket
        - Assign or reassign tickets to different team members
        - Change ticket priority, summary, or description
        - Set or update due dates
        - Add or modify labels and tags
        - Change the ticket type or other properties

        IMPORTANT: This tool requires a valid issue key. If the user mentions updating
        a ticket but doesn't provide the key, ask them which specific ticket to update.

        Args:
            issue_key: The ticket ID (REQUIRED) - format like "SCRUM-123", "AI-456", "LUNA-789"
            summary: New title/summary for the ticket (optional)
            description_text: New or updated description content (optional)
            assignee_email: Email to assign ticket to, or "unassigned" to remove assignee (optional)
            priority_name: "High", "Medium", or "Low" priority level (optional)
            due_date: Due date in YYYY-MM-DD format like "2024-12-31" (optional)
            start_date: Start date (currently not supported - parameter ignored)
            issue_type_name: "Task", "Bug", "Story", "Epic" to change ticket type (optional)
            labels: Comma-separated labels like "urgent,frontend,api" (optional)

        Returns:
            dict: Updated ticket information including success status and current field values

        Examples of when to use this tool:
            - User says: "Update ticket SCRUM-123 priority to High" → Use this tool
            - User says: "Assign SCRUM-456 to john@company.com" → Use this tool
            - User says: "Change summary of AI-789 to include new requirements" → Use this tool
            - User says: "Set due date for LUNA-100 to 2024-12-31" → Use this tool with due_date
            - User says: "Add labels 'urgent,critical' to ticket SCRUM-200" → Use this tool
            - User says: "Make SCRUM-300 unassigned" → Use this tool with assignee_email="unassigned"

        Do NOT use this tool if:
            - User doesn't provide a specific issue key
            - User wants to create a new ticket (use create_issue_sync instead)
        """
        try:
            # Convert labels string to list if provided
            labels_list = None
            if labels:
                labels_list = [label.strip() for label in labels.split(",")]

            # Log if start date was requested but inform that it's not supported
            if start_date:
                logger.info(
                    f"Start date requested ({start_date}) but start date updates are currently not supported"
                )

            result = self.utils.update_issue(
                issue_key=issue_key,
                summary=summary,
                description_text=description_text,
                assignee_email=assignee_email,
                priority_name=priority_name,
                due_date=due_date,
                start_date=None,  # Always pass None for start_date
                issue_type_name=issue_type_name,
                labels=labels_list,
            )

            return result

        except Exception as e:
            logger.error(f"Error in update_issue_sync: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to update Jira issue: {str(e)}",
            }

    def delete_issue_sync(self, issue_key: str) -> dict:
        """
        Delete an existing Jira ticket/issue permanently. Use this tool when the user wants to:
        - Remove a ticket that was created by mistake
        - Delete duplicate or invalid tickets
        - Clean up test tickets or tickets that are no longer needed
        - Remove tickets that were created in error

        WARNING: This action is PERMANENT and cannot be undone. Use with caution and
        only when the user explicitly requests deletion.

        IMPORTANT: This tool requires a valid issue key. If the user wants to delete
        a ticket but doesn't specify which one, ask them for the specific ticket key.

        Args:
            issue_key: The ticket ID to delete (REQUIRED) - format like "SCRUM-123", "AI-456", "LUNA-789"

        Returns:
            dict: Contains success status and information about the deleted ticket

        Examples of when to use this tool:
        - User says: "Delete ticket SCRUM-123" → Use this tool
        - User says: "Remove the duplicate ticket AI-456" → Use this tool
        - User says: "Delete issue LUNA-789 as it's no longer needed" → Use this tool
        - User says: "Remove SCRUM-100, it was created by mistake" → Use this tool

        Do NOT use this tool if:
        - User doesn't provide a specific issue key
        - User wants to update or modify a ticket (use update_issue_sync instead)
        - User is unclear about which ticket to delete

        Always confirm the ticket key before deletion as this cannot be undone.
        """
        try:
            result = self.utils.delete_issue(issue_key)
            return result

        except Exception as e:
            logger.error(f"Error in delete_issue_sync: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to delete Jira issue: {str(e)}",
            }

    async def process_query(
        self, user_query: UserQuery, channel_id: str = None
    ) -> dict:
        """Process Jira query using supervisor to route to appropriate agent."""
        try:
            print(f"Channel ID: {channel_id}")
            logger.info(f"Processing query from channel: {channel_id}")

            # Extract chat history for context
            chat_history_string = self.utils.extract_chat(channel_id)

            # Prepare input for supervisor
            input_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"User Query: {user_query.query}\n\nChat History: {chat_history_string}",
                    }
                ]
            }

            # Let supervisor decide which agent to use and execute
            result = self.app.invoke(input_data)

            if result and "messages" in result and len(result["messages"]) > 0:
                final_message = result["messages"][-1]
                response_content = final_message.content

                # Format the response for Slack markdown
                formatted_response = self.utils.format_for_slack(response_content)

                return {
                    "success": True,
                    "message": "Jira operation completed",
                    "data": formatted_response,
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
