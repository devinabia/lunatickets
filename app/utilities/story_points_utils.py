# utils_jira_slack.py
import os
import json
import requests
from requests.auth import HTTPBasicAuth

from jira import JIRA
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from rapidfuzz import process, fuzz
from dotenv import load_dotenv
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Define accounts OUTSIDE the class
# -------------------------------
if os.getenv("ORG") == "INABIA":
    JIRA_ACCOUNTS = {
        "default": {
            "name": "Default Account",
            "base_url": os.getenv("JIRA_BASE_URL"),
            "email": os.getenv("JIRA_EMAIL"),
            "token": os.getenv("JIRA_TOKEN"),
            "project_key": os.getenv("Default_Project"),
            "description": "Main Jira workspace for general projects",
        },
        "ark": {
            "name": "ARK Account",
            "base_url": os.getenv("JIRA_ARK_BASE_URL"),
            "email": os.getenv("JIRA_ARK_EMAIL"),
            "token": os.getenv("JIRA_ARK_TOKEN"),
            "project_key": os.getenv("JIRA_ARK_PROJECT", "AB"),
            "description": "ARK-specific Jira workspace",
        },
    }
else:
    JIRA_ACCOUNTS = {
        "default": {
            "name": "Default Account",
            "base_url": os.getenv("JIRA_BASE_URL"),
            "email": os.getenv("JIRA_EMAIL"),
            "token": os.getenv("JIRA_TOKEN"),
            "project_key": os.getenv("Default_Project"),
            "description": "Main Jira workspace for general projects",
        }
    }


class JiraSlackUtils:
    # class-level state
    ACCOUNTS = {}
    CURRENT_KEY = None

    JIRA = None
    AUTH = None
    HEADERS = {"Accept": "application/json"}

    SLACK = None  # WebClient

    # Cache for field mappings per account
    _FIELD_CACHE = {}

    @classmethod
    def init(cls, accounts: dict, default_key: str, slack_bot_token: str):
        """Initialize once at startup."""
        cls.ACCOUNTS = accounts or {}
        if default_key not in cls.ACCOUNTS:
            raise ValueError(f"default_key '{default_key}' not found in ACCOUNTS")
        if not slack_bot_token:
            raise ValueError("slack_bot_token is required")

        cls.SLACK = WebClient(token=slack_bot_token)
        cls.set_account(default_key)

    @classmethod
    def set_account(cls, key: str):
        """Switch active Jira account any time."""
        if key not in cls.ACCOUNTS:
            raise ValueError(f"Unknown Jira account key '{key}'")
        cfg = cls.ACCOUNTS[key]

        base_url = (cfg.get("base_url") or "").rstrip("/") + "/"
        email = cfg.get("email") or ""
        token = cfg.get("token") or ""
        if not base_url or not email or not token:
            raise ValueError(f"Account '{key}' missing base_url/email/token")

        cls.AUTH = HTTPBasicAuth(email, token)
        cls.JIRA = JIRA(server=base_url, basic_auth=(email, token))
        cls.CURRENT_KEY = key

        # Clear field cache for this account if switching
        if key in cls._FIELD_CACHE:
            logger.info(f"Using cached field mappings for account '{key}'")
        else:
            logger.info(
                f"Switched to Jira account '{key}', will discover fields on first use"
            )

    @classmethod
    def _collect_notification_data(cls, issue, notifications_by_channel):
        """
        Collect issue data for batched notifications.
        Groups issues by channel and then by user (Jira name).

        Args:
            issue: JIRA Issue object
            notifications_by_channel: Dict to collect notification data
        """
        try:
            f = issue.fields
            iss_key = issue.key

            # Get channel for this issue
            channel_id = cls.searchInJsonFile(iss_key)
            if channel_id is None:
                logger.debug(f"  ℹ️ No Slack channel mapping found for {iss_key}")
                return

            # Get assignee and reporter names
            people_to_notify = set()

            assignee_name = (
                getattr(f.assignee, "displayName", None)
                if getattr(f, "assignee", None)
                else None
            )
            reporter_name = (
                getattr(f.reporter, "displayName", None)
                if getattr(f, "reporter", None)
                else None
            )

            if assignee_name:
                people_to_notify.add(assignee_name)
            if reporter_name:
                people_to_notify.add(reporter_name)

            # If no assignee or reporter, track this separately
            if len(people_to_notify) == 0:
                people_to_notify.add("__NO_ASSIGNEE__")

            # Initialize channel dict if needed
            if channel_id not in notifications_by_channel:
                notifications_by_channel[channel_id] = {}

            # Add issue to each person's list
            for person_name in people_to_notify:
                if person_name not in notifications_by_channel[channel_id]:
                    notifications_by_channel[channel_id][person_name] = []

                notifications_by_channel[channel_id][person_name].append(issue)

        except Exception as e:
            logger.error(f"Error collecting notification data for {issue.key}: {e}")

    @classmethod
    def _send_batched_notifications(cls, notifications_by_channel):
        """
        Send batched Slack notifications grouped by channel and user.

        Args:
            notifications_by_channel: Dict of {channel_id: {jira_name: [issues]}}
        """
        if not notifications_by_channel:
            logger.info("📭 No notifications to send")
            return

        logger.info(
            f"\n📬 Sending batched notifications to {len(notifications_by_channel)} channel(s)..."
        )

        base_url = cls.ACCOUNTS[cls.CURRENT_KEY]["base_url"].rstrip("/")

        for channel_id, users_issues in notifications_by_channel.items():
            try:
                logger.info(f"\n📨 Processing channel {channel_id}...")

                # Get channel members
                user_list = cls.get_all_slack_channel_members(channel_id)

                if not user_list:
                    logger.warning(f"  ⚠️ No users found in channel {channel_id}")
                    # Send generic message for all issues
                    all_issues = []
                    for issues in users_issues.values():
                        all_issues.extend(issues)

                    issue_links = [
                        f"<{base_url}/browse/{iss.key}|{iss.key}>" for iss in all_issues
                    ]
                    msg = (
                        f"*Story Point Update Required*\n\nThe following tickets are missing story point estimates:\n"
                        + "\n".join([f"• {link}" for link in issue_links])
                    )
                    cls.post_to_thread(channel_id, msg)
                    continue

                slack_names = [u["name"] for u in user_list]

                # Build message parts for each user
                user_messages = []
                unassigned_issues = []

                for jira_name, issues in users_issues.items():
                    # Handle unassigned issues separately
                    if jira_name == "__NO_ASSIGNEE__":
                        unassigned_issues.extend(issues)
                        continue

                    # Fuzzy match Jira name to Slack name
                    best_match = process.extract(
                        jira_name, slack_names, scorer=fuzz.WRatio, limit=1
                    )

                    slack_user_id = None
                    if best_match and best_match[0][1] > 70:
                        matched_name = best_match[0][0]
                        user_data = next(
                            (u for u in user_list if u["name"] == matched_name), None
                        )
                        if user_data:
                            slack_user_id = user_data["id"]

                    # Build issue list for this user
                    issue_links = [
                        f"<{base_url}/browse/{iss.key}|{iss.key}>" for iss in issues
                    ]

                    if slack_user_id:
                        # User found in Slack - mention them
                        if len(issues) == 1:
                            user_msg = (
                                f"<@{slack_user_id}> - The following ticket requires story point estimation:\n"
                                + "\n".join([f"  • {link}" for link in issue_links])
                            )
                        else:
                            user_msg = (
                                f"<@{slack_user_id}> - The following {len(issues)} tickets require story point estimation:\n"
                                + "\n".join([f"  • {link}" for link in issue_links])
                            )
                        user_messages.append(user_msg)
                        logger.info(
                            f"  ✓ Will mention user {jira_name} for {len(issues)} issue(s)"
                        )
                    else:
                        # User not found in Slack - list without mention
                        if len(issues) == 1:
                            user_msg = (
                                f"*{jira_name}* - The following ticket requires story point estimation:\n"
                                + "\n".join([f"  • {link}" for link in issue_links])
                            )
                        else:
                            user_msg = (
                                f"*{jira_name}* - The following {len(issues)} tickets require story point estimation:\n"
                                + "\n".join([f"  • {link}" for link in issue_links])
                            )
                        user_messages.append(user_msg)
                        logger.info(
                            f"  ⚠️ User {jira_name} not found in Slack, listing {len(issues)} issue(s) without mention"
                        )

                # Build final message
                if user_messages or unassigned_issues:
                    msg_parts = ["📋 *Story Point Estimation Required*\n"]

                    if user_messages:
                        msg_parts.extend(user_messages)

                    if unassigned_issues:
                        msg_parts.append("\n*Tickets Without Assignee*")
                        msg_parts.append(
                            "The following tickets need to be assigned and estimated:"
                        )
                        unassigned_links = [
                            f"<{base_url}/browse/{iss.key}|{iss.key}>"
                            for iss in unassigned_issues
                        ]
                        msg_parts.extend([f"  • {link}" for link in unassigned_links])

                    final_msg = "\n\n".join(msg_parts)
                    cls.post_to_thread(channel_id, final_msg)

                    total_issues = sum(len(issues) for issues in users_issues.values())
                    logger.info(
                        f"  ✅ Sent batched notification for {total_issues} issue(s) to {len(users_issues)} user(s)"
                    )

            except Exception as e:
                logger.error(
                    f"Error sending batched notification to channel {channel_id}: {e}"
                )

    @classmethod
    def get_story_point_field_id(cls):
        """
        Dynamically discover Story Points field ID for current account.
        CRITICAL: Prioritize exact matches to avoid wrong field selection.
        """
        account_key = cls.CURRENT_KEY

        # Check cache first
        if (
            account_key in cls._FIELD_CACHE
            and "story_points" in cls._FIELD_CACHE[account_key]
        ):
            return cls._FIELD_CACHE[account_key]["story_points"]

        # Initialize cache for this account if needed
        if account_key not in cls._FIELD_CACHE:
            cls._FIELD_CACHE[account_key] = {}

        try:
            logger.info(
                f"Discovering Story Points field ID for account '{account_key}'..."
            )
            all_fields = cls.JIRA.fields()

            # CRITICAL: Check for EXACT "Story Points" match FIRST
            # This prevents selecting "Story point estimate" by mistake
            story_point_field = None

            # Priority 1: Exact match "Story Points"
            story_point_field = next(
                (f for f in all_fields if f.get("name") == "Story Points"), None
            )

            # Priority 2: Fall back to other common names
            if not story_point_field:
                story_point_names = [
                    "Story point estimate",
                    "Story Points (Estimate)",
                    "Story Point",
                    "Storypoints",
                ]
                story_point_field = next(
                    (f for f in all_fields if f.get("name") in story_point_names), None
                )

            if story_point_field:
                field_id = story_point_field["id"]
                field_name = story_point_field["name"]
                cls._FIELD_CACHE[account_key]["story_points"] = field_id
                logger.info(f"✓ Found Story Points field: '{field_name}' ({field_id})")
                return field_id
            else:
                logger.warning(
                    f"⚠ Story Points field not found for account '{account_key}'"
                )

                # Show available fields for debugging
                story_related = [
                    f"{f['name']} ({f['id']})"
                    for f in all_fields
                    if "story" in f["name"].lower() or "point" in f["name"].lower()
                ]
                logger.warning(f"  Available story/point fields: {story_related}")

                cls._FIELD_CACHE[account_key]["story_points"] = None
                return None

        except Exception as e:
            logger.error(f"Error discovering Story Points field: {e}")
            cls._FIELD_CACHE[account_key]["story_points"] = None
            return None

    @classmethod
    def get_custom_field_value_safe(cls, issue, field_id, default=None):
        """
        Safely retrieve custom field value with multiple fallback methods.
        This is the most reliable way to get custom field values.

        Args:
            issue: JIRA Issue object
            field_id: Custom field ID (e.g., 'customfield_10011')
            default: Default value if field not found

        Returns:
            Field value or default
        """
        if not field_id:
            return default

        try:
            # Method 1: Try raw dictionary access (most reliable)
            value = issue.raw["fields"].get(field_id)
            if value is not None:
                return value

            # Method 2: Try attribute access as fallback
            value = getattr(issue.fields, field_id, None)
            if value is not None:
                return value

            return default

        except (AttributeError, KeyError) as e:
            logger.debug(f"Field {field_id} not accessible for issue {issue.key}: {e}")
            return default
        except Exception as e:
            logger.error(
                f"Unexpected error accessing field {field_id} for {issue.key}: {e}"
            )
            return default

    @classmethod
    def get_all_slack_channel_members(cls, channel_id):
        try:
            members = []
            cursor = None

            while True:
                response = cls.SLACK.conversations_members(
                    channel=channel_id, cursor=cursor, limit=200
                )
                members.extend(response["members"])
                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

        except SlackApiError as e:
            logger.error(f"Error fetching Slack members for channel {channel_id}: {e}")
            return []

        userList = []
        for u in members:
            try:
                user = cls.SLACK.users_info(user=u)
                if user["user"]["is_bot"] is False:
                    u_data = {
                        "name": user["user"]["real_name"],
                        "id": user["user"]["id"],
                    }
                    userList.append(u_data)
            except SlackApiError as e:
                logger.warning(f"Could not fetch user info for {u}: {e}")
                continue

        return userList

    @classmethod
    def getRecentProject(cls):
        base_url = cls.ACCOUNTS[cls.CURRENT_KEY]["base_url"].rstrip("/")
        url = f"{base_url}/rest/api/3/project/recent"

        resp = requests.get(url, headers=cls.HEADERS, auth=cls.AUTH, timeout=30)
        resp.raise_for_status()
        projects = resp.json()

        output = []
        for proj in projects:
            proj_data = {"Key": proj["key"], "Name": proj["name"], "ID": proj["id"]}
            output.append(proj_data)
        return output

    @classmethod
    def get_first_board_for_project(cls, project_key):
        board_id = 0
        try:
            base_url = cls.ACCOUNTS[cls.CURRENT_KEY]["base_url"].rstrip("/")
            r = requests.get(
                f"{base_url}/rest/agile/1.0/board",
                params={"projectKeyOrId": project_key, "maxResults": 50},
                headers=cls.HEADERS,
                auth=cls.AUTH,
                timeout=30,
            )
            boards = r.json().get("values", [])
            board_id = boards[0]["id"] if boards else 0

        except Exception as e:
            logger.error(f"Error getting board for project {project_key}: {e}")

        return board_id

    @classmethod
    def get_upcoming_sprint_id(cls, board_id):
        sprint_id = 0
        try:
            base_url = cls.ACCOUNTS[cls.CURRENT_KEY]["base_url"].rstrip("/")
            r = requests.get(
                f"{base_url}/rest/agile/1.0/board/{board_id}/sprint",
                params={"state": "future", "maxResults": 50},
                headers=cls.HEADERS,
                auth=cls.AUTH,
                timeout=30,
            )

            futures = r.json().get("values", [])
            futures = sorted(
                futures,
                key=lambda s: (
                    s.get("startDate") is None,
                    s.get("startDate") or s.get("id"),
                ),
            )
            sprint_id = futures[0]["id"] if futures else 0
        except Exception as e:
            logger.error(f"Error getting upcoming sprint for board {board_id}: {e}")

        return sprint_id

    @classmethod
    def getUpComingSprintDetails(cls, sprint_id):
        """
        Use JIRA Python library with correct field ID.
        Now collects all issues and sends batched notifications by user.
        """

        # Get Story Points field ID dynamically
        sp_field_id = cls.get_story_point_field_id()

        if sp_field_id is None:
            logger.error(
                f"❌ Cannot find Story Points field for account '{cls.CURRENT_KEY}'"
            )
            return

        logger.info(
            f"🔍 Fetching issues from sprint {sprint_id} for account '{cls.CURRENT_KEY}'..."
        )
        logger.info(f"   Using Story Points field: {sp_field_id}")

        try:
            jql = f"sprint = {sprint_id} ORDER BY Rank ASC"

            # Use JIRA Python library with explicit field list including the correct custom field
            issues = cls.JIRA.search_issues(
                jql_str=jql,
                startAt=0,
                maxResults=False,  # Get all issues
                fields=f"summary,status,assignee,project,reporter,{sp_field_id}",
            )

            logger.info(f"✓ Found {len(issues)} issues")

            if len(issues) == 0:
                logger.info("   No issues found in this sprint")
                return

            logger.info("=" * 80)

            issues_without_sp = []
            issues_with_sp = []

            # Structure: {channel_id: {jira_name: [issue_objects]}}
            notifications_by_channel = {}

            for issue in issues:
                iss_key = issue.key

                # Access via raw fields dictionary (most reliable)
                story_points = issue.raw["fields"].get(sp_field_id)

                logger.info(f"📌 {iss_key}: Story Points = {story_points}")

                # Check if missing, zero, or empty
                if (
                    story_points is None
                    or story_points == 0
                    or story_points == 0.0
                    or story_points == ""
                ):
                    logger.info(f"  ❌ NEEDS ATTENTION - No story points set")
                    issues_without_sp.append(iss_key)

                    # Collect notification data instead of sending immediately
                    cls._collect_notification_data(issue, notifications_by_channel)
                else:
                    logger.info(f"  ✓ Has story points ({story_points})")
                    issues_with_sp.append(iss_key)

                logger.info("-" * 80)

            # Summary
            logger.info(f"\n📊 Sprint {sprint_id} Summary:")
            logger.info(f"   Total issues: {len(issues)}")
            logger.info(f"   With story points: {len(issues_with_sp)}")
            logger.info(f"   Without story points: {len(issues_without_sp)}")

            if issues_without_sp:
                logger.info(
                    f"   Issues needing attention: {', '.join(issues_without_sp)}"
                )

            # Send all batched notifications
            cls._send_batched_notifications(notifications_by_channel)

        except Exception as e:
            logger.error(f"❌ Error processing sprint {sprint_id}: {e}", exc_info=True)

    @classmethod
    def _notify_missing_story_points(cls, issue, sp_field_id):
        """
        Helper method to send Slack notifications for issues without story points.
        Works with JIRA Issue objects (not REST API dicts).

        Args:
            issue: JIRA Issue object
            sp_field_id: Story points field ID
        """
        try:
            f = issue.fields
            iss_key = issue.key

            # Get assignee and reporter (deduplicate to avoid double mentions)
            people_to_notify = set()  # Use set to automatically deduplicate

            assignee_name = (
                getattr(f.assignee, "displayName", None)
                if getattr(f, "assignee", None)
                else None
            )
            reporter_name = (
                getattr(f.reporter, "displayName", None)
                if getattr(f, "reporter", None)
                else None
            )

            if assignee_name:
                people_to_notify.add(assignee_name)
            if reporter_name:
                people_to_notify.add(reporter_name)

            base_url = cls.ACCOUNTS[cls.CURRENT_KEY]["base_url"].rstrip("/")
            issue_url = f"{base_url}/browse/{iss_key}"
            msg = f"This issue has no story points: <{issue_url}|{iss_key}>"
            channel_id = cls.searchInJsonFile(iss_key)

            if channel_id is not None:
                if len(people_to_notify) > 0:
                    user_list = cls.get_all_slack_channel_members(channel_id)

                    if not user_list:
                        # No users in channel, send generic message
                        msg = "⚠️ This issue has no assignee or reporter: " + msg
                        cls.post_to_thread(channel_id, msg)
                        logger.info(
                            f"  📨 General notification sent to channel {channel_id}"
                        )
                        return

                    names = [u["name"] for u in user_list]

                    # Use set to store unique Slack user IDs
                    mentioned_users = set()

                    for person_name in people_to_notify:
                        best_name_match = process.extract(
                            person_name, names, scorer=fuzz.WRatio, limit=2
                        )

                        if len(best_name_match) > 0 and best_name_match[0][1] > 70:
                            best_name_match_name = best_name_match[0][0]
                            idx = next(
                                (
                                    ix
                                    for ix, u in enumerate(user_list)
                                    if u.get("name") == best_name_match_name
                                ),
                                None,
                            )
                            if idx is not None:
                                mention_id = user_list[idx]["id"]
                                mentioned_users.add(
                                    mention_id
                                )  # Set automatically deduplicates

                    if len(mentioned_users) > 0:
                        # Convert set to sorted list for consistent order, then format mentions
                        # Add space after comma for better readability with multiple users
                        user_mentions = ", ".join(
                            [f"<@{uid}>" for uid in sorted(mentioned_users)]
                        )
                        msg = f"Hi {user_mentions} can you check this?\n" + msg
                        cls.post_to_thread(channel_id, msg)
                        logger.info(
                            f"  📨 Notification sent to channel {channel_id} (mentioned {len(mentioned_users)} user(s))"
                        )
                    else:
                        # Users exist in Jira but not found in Slack channel
                        msg = f"⚠️ Assignee/Reporter not found in Slack channel: " + msg
                        cls.post_to_thread(channel_id, msg)
                        logger.info(
                            f"  📨 Generic notification sent (users not found in channel)"
                        )
                else:
                    msg = "⚠️ This issue has no assignee or reporter: " + msg
                    cls.post_to_thread(channel_id, msg)
                    logger.info(
                        f"  📨 General notification sent to channel {channel_id}"
                    )
            else:
                logger.debug(f"  ℹ️ No Slack channel mapping found for {iss_key}")

        except Exception as e:
            logger.error(f"Error sending notification for {issue.key}: {e}")

    @classmethod
    def post_to_thread(cls, channel_id, txt):
        """Send message to Slack channel."""
        try:
            cls.SLACK.chat_postMessage(channel=channel_id, text=txt)
        except SlackApiError as e:
            logger.error(f"Error posting to Slack channel {channel_id}: {e}")

    @classmethod
    def searchInJsonFile(cls, issue_key):
        """Search for Slack channel ID associated with issue key."""
        issue_key = issue_key.lower().strip()
        channel_id = None

        try:
            with open("slack_message.json", "r", encoding="utf-8") as kk:
                data = json.load(kk)

                idx, rec = next(
                    (
                        (i, r)
                        for i, r in enumerate(data)
                        if str(r.get("issue_key", "")).casefold() == issue_key
                    ),
                    (None, None),
                )

                if rec:
                    channel_id = rec["channel_id"]
        except FileNotFoundError:
            logger.warning("slack_message.json file not found")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing slack_message.json: {e}")
        except Exception as e:
            logger.error(f"Error searching JSON file: {e}")

        return channel_id
