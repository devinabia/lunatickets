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
        assignee_email: str,
        priority_name: str = None,
        reporter_email: str = None,
        issue_type_name: str = None,
        force_update_description_after_create: bool = True,
    ) -> dict:
        """Create Jira issue - main function used by the create agent."""
        if issue_type_name is None:
            issue_type_name = self.default_issue_type

        # Use default project if not provided or empty
        if not project_name_or_key or project_name_or_key.strip() == "":
            project_name_or_key = self.default_project

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

            # Set assignee
            if "assignee" in allowed and assignee_email:
                try:
                    assignee_id = self.utils.get_account_id(assignee_email)
                    fields["assignee"] = {"id": assignee_id}
                except Exception as e:
                    logger.warning(f"Could not set assignee '{assignee_email}': {e}")

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
                "priority": (final["fields"]["priority"] or {}).get("name"),
                "assignee": (final["fields"]["assignee"] or {}).get("displayName"),
                "status": (final["fields"]["status"] or {}).get("name"),
                "url": f"{self.base_url.rstrip('/')}/browse/{issue_key}",
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

    def update_issue_sync(
        self,
        issue_key: str,
        summary: str = None,
        description_text: str = None,
        assignee_email: str = None,
        priority_name: str = None,
        due_date: str = None,
        issue_type_name: str = None,
        labels: str = None,  # Comma-separated string
    ) -> dict:
        """Update Jira issue - function used by the update agent."""
        try:
            # Convert labels string to list if provided
            labels_list = None
            if labels:
                labels_list = [label.strip() for label in labels.split(",")]

            result = self.utils.update_issue(
                issue_key=issue_key,
                summary=summary,
                description_text=description_text,
                assignee_email=assignee_email,
                priority_name=priority_name,
                due_date=due_date,
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
        """Delete Jira issue - function used by the delete agent."""
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
