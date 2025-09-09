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
        self.default_project = "AI"

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

        # Create specialized agents with simple sprint list tool
        self.create_agent = create_react_agent(
            model=self.model,
            tools=[
                self.create_issue_sync,
                self.get_sprint_list_sync,  # Simple sprint list tool
            ],
            name="jira_create_expert",
            prompt=Prompt.TICKET_CREATION_PROMPT,
        )

        self.update_agent = create_react_agent(
            model=self.model,
            tools=[
                self.update_issue_sync,
                self.get_sprint_list_sync,  # Sprint list for ongoing sprint detection
                self.get_project_from_issue_sync,  # Helper to get project from issue key
            ],
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

    def get_sprint_list_sync(self, project_name_or_key: str) -> dict:
        """
        Get list of sprints for a project. Use this when you need sprint options.

        Args:
            project_name_or_key: Jira project key (e.g., "SCRUM", "LUNA_TICKETS")

        Returns:
            dict: Sprint list information
        """
        try:
            project_key = self.utils.resolve_project_key(project_name_or_key)
            sprint_list = self.utils.get_all_sprints_for_project(project_key)

            return {
                "success": True,
                "project_key": project_key,
                "sprint_list": sprint_list,
            }
        except Exception as e:
            logger.error(f"Error getting sprint list: {e}")
            return {
                "success": False,
                "error": str(e),
                "sprint_list": f"Could not get sprints for {project_name_or_key}. Use 'backlog'.",
            }

    def get_default_sprint_for_project(self, project_key: str) -> dict:
        """
        Get the default sprint for a project (the immediate next sprint after ongoing).
        Uses sprint name parsing to determine chronological order.
        """
        try:
            # Use your existing method to get sprint list
            sprint_info = self.get_sprint_list_sync(project_key)

            if not sprint_info["success"]:
                logger.warning(f"Could not get sprints for {project_key}")
                return {
                    "has_default": False,
                    "sprint_name": None,
                    "ask_user": True,
                    "sprint_list": f"Could not get sprints for {project_key}. Use 'backlog'.",
                }

            sprint_list = sprint_info["sprint_list"]

            # Get the actual sprint data using your existing utils method
            board_id = self.utils._get_board_id_for_project(project_key)
            if not board_id:
                logger.warning(f"No board found for project {project_key}")
                return {
                    "has_default": False,
                    "sprint_name": None,
                    "ask_user": True,
                    "sprint_list": sprint_list,
                }

            # Get all active and future sprints with full details
            r = self.session.get(
                f"{self.base_url}/rest/agile/1.0/board/{board_id}/sprint",
                params={"state": "active,future", "maxResults": 50},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()

            ongoing_sprint = None
            future_sprints = []

            # Separate ongoing and future sprints
            for sprint in data.get("values", []):
                if sprint["state"] == "active":
                    ongoing_sprint = sprint
                elif sprint["state"] == "future":
                    future_sprints.append(sprint)

            if not future_sprints:
                logger.info(f"No future sprints found for {project_key}, will ask user")
                return {
                    "has_default": False,
                    "sprint_name": None,
                    "ask_user": True,
                    "sprint_list": sprint_list,
                }

            # Custom sorting function for sprint names like "AI -- W36-Y25", "AI -- W37-Y26"
            def extract_week_number(sprint_name):
                """Extract week number from sprint name for sorting."""
                try:
                    # Look for pattern like "W36" in the name
                    import re

                    match = re.search(r"W(\d+)", sprint_name)
                    if match:
                        return int(match.group(1))

                    # Fallback: try to extract any number from the name
                    numbers = re.findall(r"\d+", sprint_name)
                    if numbers:
                        return int(numbers[0])

                    # Final fallback: use sprint ID
                    return 0
                except:
                    return 0

            # Sort future sprints by week number extracted from name
            future_sprints.sort(key=lambda x: extract_week_number(x["name"]))

            # Log the sorting for debugging
            logger.info(f"Future sprints sorted order:")
            for sprint in future_sprints:
                week_num = extract_week_number(sprint["name"])
                logger.info(f"  - {sprint['name']} (week {week_num})")

            # The first future sprint after sorting is the immediate next one
            next_sprint = future_sprints[0]

            logger.info(
                f"Using immediate next sprint as default: {next_sprint['name']}"
            )
            return {
                "has_default": True,
                "sprint_name": next_sprint["name"],
                "ask_user": False,
                "sprint_list": sprint_list,
            }

        except Exception as e:
            logger.error(f"Error getting default sprint for {project_key}: {e}")
            return {
                "has_default": False,
                "sprint_name": None,
                "ask_user": True,
                "sprint_list": f"Could not get sprints for {project_key}. Use 'backlog'.",
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
        force_update_description_after_create: bool = True,
    ) -> dict:
        """
        Create a new Jira ticket/issue.

        CRITICAL REQUIREMENT: assignee_email is MANDATORY and must not be empty.

        This function will FAIL and return an error if no assignee is provided.
        Do NOT call this function unless you have a valid assignee_email.

        If the user doesn't specify who to assign the ticket to, ask them first:
        "Who should I assign this ticket to?"

        Args:
            assignee_email: REQUIRED - Who to assign to (cannot be empty)
            project_name_or_key: Project key (optional - defaults to "AI")
            summary: Ticket title (optional - auto-generated if empty)
            description_text: Ticket description (optional - auto-generated if empty)
            priority_name: Priority level (optional)
            reporter_email: Who reported it (optional)
            issue_type_name: Issue type (optional - defaults to "Task")
            sprint_name: Sprint name or None for backlog (optional)
            force_update_description_after_create: Whether to force description update (optional)

        Returns:
            dict: Either success with ticket details OR error asking for assignee

        Examples:
            ✅ CORRECT USAGE:
            - User: "create ticket assign to john" → call with assignee_email="john"
            - User: "make story for sarah" → call with assignee_email="sarah"

            ❌ INCORRECT USAGE:
            - User: "create story" → DON'T call this function, ask for assignee instead
            - User: "make ticket" → DON'T call this function, ask for assignee instead
        """
        if issue_type_name is None:
            issue_type_name = self.default_issue_type

        # Use default project if not provided
        if not project_name_or_key or project_name_or_key.strip() == "":
            project_name_or_key = self.default_project
            logger.info(f"No project specified, using default: {self.default_project}")

        # Handle sprint selection with user fallback
        if not sprint_name:
            sprint_info = self.get_default_sprint_for_project(project_name_or_key)

            if sprint_info["has_default"]:
                # Use the upcoming sprint as default
                sprint_name = sprint_info["sprint_name"]
                logger.info(f"Using default upcoming sprint: {sprint_name}")
            else:
                # No upcoming sprint available - ask user to choose
                logger.info(
                    f"No upcoming sprint available for {project_name_or_key}, asking user"
                )
                return {
                    "success": False,
                    "needs_sprint_selection": True,
                    "message": f"Which sprint should I add this ticket to? Here are the available options:\n\n{sprint_info['sprint_list']}\n\nPlease specify the exact sprint name or 'backlog'.",
                    "project": project_name_or_key,
                    "assignee": assignee_email,
                    "sprint_options": sprint_info["sprint_list"],
                }

        # Generate default summary if empty
        if not summary or summary.strip() == "":
            summary = "New ticket created via AI assistant"

        # Generate default description if empty
        if not description_text or description_text.strip() == "":
            description_text = "This ticket was created through the AI assistant and needs further details to be added."

        # Rest of your existing create_issue_sync code remains exactly the same...
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
                else:
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
                fields["labels"] = ["created-by-luna"]

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

            # SPRINT HANDLING
            sprint_status = "Backlog"

            if sprint_name and sprint_name.strip():
                try:
                    board_id = self.utils._get_board_id_for_project(project_key)
                    if not board_id:
                        raise RuntimeError(f"No board found for project {project_key}")
                    sprint_id = self.utils._get_sprint_id_by_name(
                        board_id, sprint_name.strip()
                    )
                    if not sprint_id:
                        raise RuntimeError(
                            f"{sprint_name.strip()} not found on this board"
                        )
                    self.utils._add_issue_to_sprint(sprint_id, issue_key)
                    sprint_status = sprint_name.strip()
                    logger.info(
                        f"Issue {issue_key} moved to sprint {sprint_name.strip()} (id={sprint_id})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to move issue to sprint {sprint_name}: {e}")
                    sprint_status = f"Backlog (failed to move to {sprint_name.strip()})"

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
                "sprint": sprint_status,
                "project": project_key,
            }

            # Add assignment information if there were issues
            if assignment_info["suggestions"]:
                result["assignment_failed"] = True
                result["user_suggestions"] = assignment_info["suggestions"]
                result["assignment_message"] = (
                    f"Ticket created but could not assign to '{assignee_email}'"
                )

            logger.info(
                f"Issue {issue_key} created in project {project_key} with status: {result.get('status')} in sprint: {sprint_status}"
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
        sprint_name: str = None,  # Sprint movement support with ongoing detection
        status_name: str = None,  # NEW: Status update parameter
    ) -> dict:
        """
        Update an existing Jira ticket/issue. Use this tool when the user wants to:
        - Modify any field of an existing ticket
        - Assign or reassign tickets to different team members
        - Change ticket priority, summary, or description
        - Set or update due dates
        - Add or modify labels and tags
        - Change the ticket type or other properties
        - Move ticket to different sprint or backlog
        - Update ticket status (To Do, In Progress, Done, etc.)

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
            sprint_name: Specific sprint name or "backlog" to move ticket (optional)
            status_name: Status to transition to like "In Progress", "Done", "To Do" (optional)

        Returns:
            dict: Updated ticket information including success status and current field values
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
                sprint_name=sprint_name,  # Pass sprint name for movement
                status_name=status_name,  # NEW: Pass status name for updates
            )

            return result

        except Exception as e:
            logger.error(f"Error in update_issue_sync: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to update Jira issue: {str(e)}",
            }

    def get_project_from_issue_sync(self, issue_key: str) -> dict:
        """
        Helper tool to get the project key from an issue key.
        Use this when you need to know which project an issue belongs to.

        Args:
            issue_key: The ticket ID (e.g., "LT-23", "SCRUM-456")

        Returns:
            dict: Project information for the issue
        """
        try:
            issue_data = self.utils.get_issue(issue_key, "project")
            project_key = issue_data["fields"]["project"]["key"]

            return {"success": True, "issue_key": issue_key, "project_key": project_key}
        except Exception as e:
            logger.error(f"Error getting project for issue {issue_key}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Could not find project for issue {issue_key}",
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
