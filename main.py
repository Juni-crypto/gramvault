"""InstaIntel — Main Entry Point

Starts the Telegram bot for:
1. Receiving Instagram URLs → download, analyze, index
2. Querying saved content via Claude RAG

Usage:
    python main.py
"""

import click

from config import Config
from utils.logger import log


@click.command()
def main():
    """Start InstaIntel Telegram bot."""

    Config.validate()
    Config.ensure_dirs()

    log.info("=" * 60)
    log.info("InstaIntel -- Starting up")
    log.info("=" * 60)

    from bot.telegram_bot import start_bot
    start_bot()


if __name__ == "__main__":
    main()
