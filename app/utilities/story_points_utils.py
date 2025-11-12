# utils_jira_slack.py
import os
import json
import requests
from requests.auth import HTTPBasicAuth

from jira import JIRA
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from rapidfuzz import process, fuzz


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

    # ------------- your functions (unchanged names) -------------

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

            print(f"Total members in channel: {len(members)}")

        except SlackApiError as e:
            print(f"Error fetching members: {e.response['error']}")
            return []

        userList = []
        for u in members:
            user = cls.SLACK.users_info(user=u)
            if user["user"]["is_bot"] is False:
                u_data = {"name": user["user"]["real_name"], "id": user["user"]["id"]}
                userList.append(u_data)

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
    def getStoryPoint(cls, issue_key):
        try:
            issue = cls.JIRA.issue(issue_key)

            sp_id = next(
                (
                    f.get("id")
                    for f in cls.JIRA.fields()
                    if f.get("name") in ("Story Points", "Story point estimate")
                ),
                None,
            )

            if sp_id is not None:
                story_points = getattr(
                    issue.fields, sp_id, None
                )  # or: issue.raw["fields"][sp_id]
                return story_points
            else:
                return None

        except Exception as e:
            print(f"Error retrieving Jira issue {issue_key}: {e}")
            return None

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
            print("Error here-2:", e)

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
            print("Error here-1:", e)

        return sprint_id

    @classmethod
    def getUpComingSprintDetails(cls, sprint_id):
        fields = ["summary", "status", "assignee", "project", "reporter"]
        jql = "sprint = " + str(sprint_id) + " ORDER BY Rank ASC"
        issues = cls.JIRA.search_issues(
            jql_str=jql,
            startAt=0,
            maxResults=False,  # fetch all via internal pagination
            fields=",".join(fields),
        )

        for i in issues:
            f = i.fields
            assignee_reporter_list = []
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
            iss_key = i.key

            base_url = cls.ACCOUNTS[cls.CURRENT_KEY]["base_url"].rstrip("/")
            issue_url = f"{base_url}/browse/{iss_key}"

            if cls.getStoryPoint(iss_key) is None:
                if assignee_name is None:
                    assignee_name = ""
                if reporter_name is None:
                    reporter_name = ""

                if len(assignee_name) > 0:
                    assignee_reporter_list.append(assignee_name)
                if len(reporter_name) > 0:
                    assignee_reporter_list.append(reporter_name)

                msg = (
                    "This issue key has not updated story points: "
                    f"<{issue_url}|{iss_key}>"
                )
                channel_id = cls.searchInJsonFile(iss_key)

                if channel_id is not None:

                    if len(assignee_reporter_list) > 0:
                        user_id = ""
                        for arl in assignee_reporter_list:
                            user_list = cls.get_all_slack_channel_members(channel_id)
                            names = [u["name"] for u in user_list]

                            if len(user_list) > 0:
                                best_name_match = process.extract(
                                    arl, names, scorer=fuzz.WRatio, limit=2
                                )

                                if len(best_name_match) > 0:
                                    score = best_name_match[0][1]

                                    if score > 70:
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
                                            user_id = f"<@{mention_id}>" + "," + user_id
                        if len(user_id) > 0:
                            msg = f"Hi {user_id} can you check this?\n" + msg
                            cls.post_to_thread(channel_id, msg)
                    else:
                        # general msg , when no reporter and no assignee
                        msg = (
                            "This is general message when no reporter and no assignee found "
                            + msg
                        )
                        cls.post_to_thread(channel_id, msg)

    @classmethod
    def post_to_thread(cls, channel_id, txt):
        cls.SLACK.chat_postMessage(channel=channel_id, text=txt)

    @classmethod
    def searchInJsonFile(cls, issue_key):
        issue_key = issue_key.lower().strip()
        channel_id = None
        with open("slack_message.json", "r+", encoding="utf-8") as kk:
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
                print(rec)
                channel_id = rec["channel_id"]

        return channel_id
