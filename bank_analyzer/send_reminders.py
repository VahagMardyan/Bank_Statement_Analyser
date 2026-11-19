"""Send today's reminder emails. Meant to run once a day (Windows Task Scheduler).

    python send_reminders.py              send for real (SMTP settings from .env)
    python send_reminders.py --dry-run    print the emails, send and record nothing
    python send_reminders.py --dry-run --today 2026-11-28   pretend it's another day
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from modules.facade import BankAnalyzerFacade
from modules.i18n import Translator
from modules.notifications import (
    ConsoleChannel,
    EmailChannel,
    MySqlNotificationStore,
    NotificationStore,
    ReminderService,
)

BASE_DIR = Path(__file__).resolve().parent


class DryRunStore(NotificationStore):
    """Reads like the real store but never writes, so a dry run can be repeated."""

    def __init__(self, real: NotificationStore) -> None:
        self.real = real

    def recipients(self):
        return self.real.recipients()

    def already_sent(self, user_id, kind, keys):
        return self.real.already_sent(user_id, kind, keys)

    def mark_sent(self, user_id, kind, keys):
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print instead of sending")
    parser.add_argument("--today", type=date.fromisoformat, help="YYYY-MM-DD, for testing")
    args = parser.parse_args()

    store: NotificationStore = MySqlNotificationStore()
    if args.dry_run:
        channel, store = ConsoleChannel(), DryRunStore(store)
    else:
        try:
            channel = EmailChannel.from_env()
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

    service = ReminderService(
        store=store,
        channel=channel,
        translator=Translator.from_directory(BASE_DIR / "locales"),
        facade_for_user=BankAnalyzerFacade.for_user,
        today=args.today,
    )
    sent = service.run()
    print(f"{service.today.isoformat()}: {len(sent)} email(s)" + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
