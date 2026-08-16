"""
ElasticCap Core: Sink-Authority Budget Algebra

Implements the formal capability algebra from ChainCaps:
- Atomic sink privileges with partial ordering
- Budgets as downward-closed subsets (ideals)
- Monotonic meet-propagation across tool chains
- Non-amplification invariant enforcement

This module is identical to ChainCaps' budget.py — the budget algebra
itself does not change in ElasticCap.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import FrozenSet, Optional, Set
from urllib.parse import unquote


class SinkType(Enum):
    """Atomic sink operation types."""
    DISPLAY = auto()
    SEND_HTTP = auto()
    SEND_EMAIL = auto()
    WRITE_FILE = auto()
    EXECUTE = auto()
    DB_WRITE = auto()
    MEMORY_WRITE = auto()


@dataclass(frozen=True)
class SinkPrivilege:
    """An atomic sink privilege: (operation, resource_scope).

    Examples:
        SinkPrivilege(SEND_HTTP, "api.example.com")
        SinkPrivilege(WRITE_FILE, "/tmp/*")
        SinkPrivilege(DISPLAY, "*")  # unrestricted display
    """
    operation: SinkType
    scope: str = "*"  # "*" means unrestricted within this operation

    @staticmethod
    def _canonicalize_scope(scope: str) -> str:
        decoded = unquote(scope)
        if decoded.startswith("/"):
            decoded = os.path.normpath(decoded)
        return decoded

    def subsumes(self, other: SinkPrivilege) -> bool:
        if self.operation != other.operation:
            return False
        if self.scope == "*":
            return True

        canon_self = self._canonicalize_scope(self.scope)
        canon_other = self._canonicalize_scope(other.scope)

        if self.operation in (SinkType.SEND_HTTP,):
            canon_self = self._strip_protocol(canon_self)
            canon_other = self._strip_protocol(canon_other)

        if canon_self == canon_other:
            return True
        if canon_self.endswith("*"):
            prefix = canon_self[:-1]
            return canon_other.startswith(prefix)
        if canon_self.startswith("@") and canon_other.endswith(canon_self):
            return True
        return False

    @staticmethod
    def _strip_protocol(scope: str) -> str:
        for prefix in ("https://", "http://"):
            if scope.startswith(prefix):
                return scope[len(prefix):]
        return scope

    def __repr__(self) -> str:
        return f"{self.operation.name.lower()}({self.scope})"


@dataclass(frozen=True)
class Budget:
    """A sink-authority budget: downward-closed set of authorized sink privileges."""
    privileges: FrozenSet[SinkPrivilege]

    @staticmethod
    def top(sink_types: Optional[Set[SinkType]] = None) -> Budget:
        if sink_types is None:
            sink_types = set(SinkType)
        return Budget(frozenset(
            SinkPrivilege(op, "*") for op in sink_types
        ))

    @staticmethod
    def bottom() -> Budget:
        return Budget(frozenset())

    @staticmethod
    def from_sinks(*privs: SinkPrivilege) -> Budget:
        return Budget(frozenset(privs))

    def meet(self, other: Budget) -> Budget:
        result = set()
        for p in self.privileges:
            for q in other.privileges:
                if q.subsumes(p):
                    result.add(p)
                    break
        for p in other.privileges:
            for q in self.privileges:
                if q.subsumes(p):
                    result.add(p)
                    break
        return Budget(frozenset(result))

    def authorizes(self, priv: SinkPrivilege) -> bool:
        for p in self.privileges:
            if p.subsumes(priv):
                return True
        return False

    def is_subset_of(self, other: Budget) -> bool:
        for p in self.privileges:
            if not other.authorizes(p):
                return False
        return True

    def __le__(self, other: Budget) -> bool:
        return self.is_subset_of(other)

    def __repr__(self) -> str:
        if not self.privileges:
            return "Budget(EMPTY)"
        privs = sorted(str(p) for p in self.privileges)
        return f"Budget({', '.join(privs)})"


# --- Common budget factories ---

def budget_display_only() -> Budget:
    return Budget.from_sinks(SinkPrivilege(SinkType.DISPLAY))


def budget_internal_email(scope: str = "@corp.com") -> Budget:
    return Budget.from_sinks(
        SinkPrivilege(SinkType.DISPLAY),
        SinkPrivilege(SinkType.SEND_EMAIL, scope),
    )


def budget_public() -> Budget:
    return Budget.top()


def budget_sensitive_file() -> Budget:
    return Budget.from_sinks(
        SinkPrivilege(SinkType.DISPLAY),
        SinkPrivilege(SinkType.WRITE_FILE, "/tmp/*"),
    )
