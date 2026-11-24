"""Copies community rules into config/category_rules.json.

A community rule is a correction enough users agree on: at least 3 users
mapped the same keyword to the same category, and at least 75% of the users
who corrected that keyword chose it (COMMUNITY_MIN_USERS and
COMMUNITY_MIN_SHARE in .env change this). The app already applies these rules
live from the database. This script puts them into the shared JSON file so
they stay in git, also for runs without a database.

    python promote_rules.py            # show what would change, change nothing
    python promote_rules.py --write    # change config/category_rules.json

It reads the database from .env (local MySQL). For TiDB, set the DB_* values
in the PowerShell session first, as described in the README. Check the
result with `git diff config/category_rules.json` before committing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modules.mysql_repositories import MySqlCommunityRules
from modules.rules import DEFAULT_RULES_PATH, LABEL_SEPARATOR, CommunityRule
from modules.text_utils import DEFAULT_NORMALIZER

norm = DEFAULT_NORMALIZER.clean


def category_key(data: dict, category: str) -> str | None:
    """JSON key for a category ("Transport" also finds "Transport | Տրանսպորտ")."""
    for key in data:
        if key.split(LABEL_SEPARATOR, 1)[0].strip() == category:
            return key
    return None


def keywords_of(entry) -> list:
    """The keyword list of one category, in the new or the old list form."""
    if isinstance(entry, list):
        return entry
    return entry.setdefault("keywords", [])


def merchants_of(entry) -> list[tuple[str, list[str]]]:
    """``(merchant name, normalized aliases)`` of one category."""
    if not isinstance(entry, dict):
        return []
    out = []
    for merchant in entry.get("merchants", []):
        if isinstance(merchant, str):
            out.append((merchant, [norm(merchant)]))
        elif isinstance(merchant, dict):
            name = str(merchant.get("name", ""))
            aliases = [name, *(str(a) for a in merchant.get("aliases", []))]
            out.append((name, [norm(a) for a in aliases if a]))
    return out


def promote(data: dict, rules: list[CommunityRule]) -> list[str]:
    """Apply the rules to ``data`` in place; return report lines."""
    report = []
    for rule in rules:
        votes = f"{rule.users}/{rule.total_users} users"
        key = category_key(data, rule.category)
        if key is None:
            report.append(f"skip   '{rule.keyword}': category {rule.category} is not in the file")
            continue
        target = keywords_of(data[key])
        if rule.keyword in (norm(k) for k in target):
            report.append(f"same   '{rule.keyword}' -> {rule.category} (already there, {votes})")
            continue

        for other_key, entry in data.items():
            if other_key == key:
                continue
            words = keywords_of(entry)
            kept = [k for k in words if norm(k) != rule.keyword]
            if len(kept) != len(words):
                words[:] = kept
                report.append(f"move   '{rule.keyword}': removed from {other_key}")
            for name, aliases in merchants_of(entry):
                if any(a and (a in rule.keyword or rule.keyword in a) for a in aliases):
                    report.append(
                        f"check  '{rule.keyword}': merchant '{name}' in {other_key} matches "
                        "the same text. In the JSON, merchant rules come before keywords, "
                        "so move or fix that merchant by hand"
                    )
        target.append(rule.keyword)
        report.append(f"add    '{rule.keyword}' -> {rule.category} ({votes})")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="change the JSON file")
    parser.add_argument("--min-users", type=int, help="override COMMUNITY_MIN_USERS")
    parser.add_argument("--min-share", type=float, help="override COMMUNITY_MIN_SHARE (0-1)")
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES_PATH, help="JSON file")
    args = parser.parse_args()

    rules = MySqlCommunityRules(min_users=args.min_users, min_share=args.min_share).load()
    if not rules:
        print("No community rules yet: no keyword has enough users agreeing.")
        return

    data = json.loads(args.rules.read_text(encoding="utf-8"))
    for line in promote(data, rules):
        print(line)

    if args.write:
        args.rules.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved {args.rules}. Check it with: git diff {args.rules.as_posix()}")
    else:
        print("\nNothing was changed. Run with --write to save.")


if __name__ == "__main__":
    main()
