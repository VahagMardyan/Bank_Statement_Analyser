"""Shared application constants."""

from __future__ import annotations

DEFAULT_BUDGETS: dict[str, int] = {
    "Supermarket": 100000,
    "Transport": 30000,
    "Cafes and Restaurants": 50000,
    "Utilities": 40000,
    "Shopping": 50000,
    "Fuel": 40000,
    "Subscriptions": 15000,
    "Healthcare": 30000,
    "Fitness and Sport": 25000,
    "Entertainment": 25000,
    "Beauty and Personal Care": 20000,
    "E-commerce": 30000,
    "Education": 50000,
    "Pets": 15000,
    "Travel": 100000,
    "Insurance": 20000,
    "Loans and Credit": 100000,
    "Rent": 150000,
    "Charity and Donations": 10000,
    "Government and Fees": 15000,
    "Fee": 5000,
    "Other": 20000,
}

INCOME_CATEGORIES: frozenset[str] = frozenset({"Income", "Salary"})
