"""Monthly budget limits per category: the storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

class BudgetRepository(ABC):
    @abstractmethod
    def load(self) -> dict[str, float]:
        """Saved limits by category; categories never saved are missing."""

    @abstractmethod
    def save(self, budgets: dict[str, float]) -> None:
        pass

