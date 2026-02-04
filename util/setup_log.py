import logging
import os
import atexit
import asyncio
from telegram import Bot
from dotenv import load_dotenv


class TelegramBotHandler(logging.Handler):
    def __init__(self, token, chat_id, topic_id=None):
        super().__init__()
        self.token = token
        self.chat_id = chat_id
        self.topic_id = topic_id

    def emit(self, record):
        log_entry = self.format(record)
        try:
            # Check if there is a running loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():

                async def send_async():
                    async with Bot(token=self.token) as bot:
                        await bot.send_message(
                            chat_id=self.chat_id,
                            text=log_entry,
                            message_thread_id=self.topic_id,
                        )

                loop.create_task(send_async())
            else:
                # Sync context: Use asyncio.run with context manager to ensure cleanup
                async def send_sync():
                    async with Bot(token=self.token) as bot:
                        await bot.send_message(
                            chat_id=self.chat_id,
                            text=log_entry,
                            message_thread_id=self.topic_id,
                        )

                asyncio.run(send_sync())
        except Exception:
            self.handleError(record)


def setup_log():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("app.log", mode="a"),
            logging.StreamHandler(),  # Console output
        ],
    )

    telegram_token = os.getenv("TELEGRAM_TOKEN")
    telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
    telegram_topic_id = os.getenv("TELEGRAM_TOPIC_ID")

    if telegram_token and telegram_chat_id:
        telegram_handler = TelegramBotHandler(
            token=telegram_token, chat_id=telegram_chat_id, topic_id=telegram_topic_id
        )
        telegram_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        logging.getLogger().addHandler(telegram_handler)

        def on_exit():
            logging.info("Program exited.")

        atexit.register(on_exit)

    # Suppress httpx and telegram logs to avoid noise and potential infinite loops
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return logging
