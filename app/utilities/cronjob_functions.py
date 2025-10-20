import os
from dotenv import load_dotenv
import httpx
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

load_dotenv()
logger = logging.getLogger(__name__)


class Cronjob:
    scheduler = AsyncIOScheduler()

    @staticmethod
    async def dump_qdrant_data():
        if os.getenv("ENVIRONMENT") == "PROD":
            try:
                url = f"{os.getenv('APP_BACKEND_URL')}dump-data"
                async with httpx.AsyncClient() as client:
                    response = await client.post(url)
                    print(f"Qdrant data dump completed: {response.status_code}")
                    return True
            except Exception as e:
                print(f"Error dumping Qdrant data: {e}")
                return False
        else:
            print("Skipping dump - not in PROD environment")
            return False

    @staticmethod
    async def start_scheduler():
        Cronjob.scheduler.add_job(
            Cronjob.dump_qdrant_data,
            CronTrigger(hour=0, minute=0),
            id="dump_qdrant_data",
            coalesce=True,
            replace_existing=True,
        )

        Cronjob.scheduler.start()
        print("Scheduler started successfully")
