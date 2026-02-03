import logging

def setup_log():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("app.log", mode="a"),
            logging.StreamHandler(),  # Console output
        ],
    )

    return logging
