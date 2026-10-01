"""
Domain-agnostic identity, terminology, and reward configuration loader.

Loads domain definitions from ``src/config/domains/{name}.yml`` and provides
a unified API that training, eval, and data-prep scripts use instead of
importing directly from nexus_identity.py.

The ``offsec`` domain is the default for backward compatibility. When no
domain is specified, all accessors fall back to the hardcoded constants
in nexus_identity.py so that existing single-model pipelines are unaffected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_DOMAINS_DIR = Path(__file__).parent / "domains"
_REGISTRY: dict[str, DomainConfig] = {}


@dataclass(frozen=True)
class RoleConfig:
    """Configuration for a single swarm role within a domain."""

    name: str
    extends_identity: str  # "full", "short", or "none"
    directives: str
    reasoning_format: str | None = None
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    categories: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> RoleConfig:
        allowed = raw.get("allowed_tools")
        return cls(
            name=name,
            extends_identity=raw.get("extends_identity", "full"),
            directives=raw.get("directives", "").rstrip(),
            reasoning_format=raw.get("reasoning_format"),
            allowed_tools=frozenset(allowed) if allowed else frozenset(),
            categories=raw.get("categories", {}),
        )


@dataclass(frozen=True)
class DomainConfig:
    """Parsed domain definition with typed accessors."""

    _raw: dict[str, Any] = field(repr=False)

    @property
    def name(self) -> str:
        return self._raw["domain"]["name"]

    @property
    def display_name(self) -> str:
        return self._raw["domain"].get("display_name", self.name)

    @property
    def agent_name(self) -> str:
        return self._raw["persona"]["agent_name"]

    @property
    def identity(self) -> str:
        return self._raw["persona"]["identity"].rstrip()

    @property
    def identity_short(self) -> str:
        return self._raw["persona"]["identity_short"].rstrip()

    @property
    def domain_terms(self) -> frozenset[str]:
        return frozenset(self._raw["persona"].get("domain_terms", []))

    @property
    def break_phrases(self) -> list[str]:
        return list(self._raw["persona"].get("break_phrases", []))

    @property
    def skills_index(self) -> str:
        return self._raw["persona"].get("skills_index", "").rstrip()

    @property
    def subagents_index(self) -> str:
        return self._raw["persona"].get("subagents_index", "").rstrip()

    @property
    def identity_corrections(self) -> list[str]:
        return list(self._raw["persona"].get("identity_corrections", []))

    @property
    def deployment(self) -> dict[str, Any]:
        return self._raw.get("deployment", {})

    def get_role(self, role: str) -> RoleConfig:
        roles = self._raw.get("roles", {})
        if role not in roles:
            raise ValueError(
                f"Unknown role {role!r} for domain {self.name!r}. "
                f"Available: {sorted(roles)}"
            )
        return RoleConfig.from_dict(role, roles[role])

    def get_role_names(self) -> list[str]:
        return sorted(self._raw.get("roles", {}))

    def get_role_identity(self, role: str) -> str:
        """Build the composite identity string for a swarm role."""
        rc = self.get_role(role)
        if rc.extends_identity == "full":
            base = self.identity
        elif rc.extends_identity == "short":
            base = self.identity_short
        else:
            base = ""
        if base and rc.directives:
            return f"{base}\n\n{rc.directives}"
        return rc.directives or base

    def build_system_prompt(
        self,
        tools: list[dict],
        role: str | None = None,
        *,
        include_skills: bool = True,
        include_subagents: bool = True,
    ) -> str:
        """Assemble the full system prompt for a domain + role combination.

        When *role* is None, produces a single-model system prompt.
        When set, produces a role-specific prompt with tool filtering.
        """
        if role is None:
            identity = self.identity
            filtered_tools = tools
        else:
            identity = self.get_role_identity(role)
            rc = self.get_role(role)
            if rc.allowed_tools:
                filtered_tools = [
                    t for t in tools
                    if t.get("function", {}).get("name") in rc.allowed_tools
                ]
            elif role == "worker":
                filtered_tools = [
                    t for t in tools
                    if t.get("function", {}).get("name") != "Task"
                ]
            else:
                filtered_tools = tools

        parts = [identity]

        if role != "triage" and filtered_tools:
            tool_lines = "\n".join(
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in filtered_tools
            )
            parts.append(
                f"Call tools with JSON arguments matching each tool's schema:\n{tool_lines}"
            )

        show_skills = include_skills and (role is None or role == "orchestrator")
        show_subagents = include_subagents and (role is None or role == "orchestrator")

        if show_skills and self.skills_index:
            parts.append(self.skills_index)
        if show_subagents and self.subagents_index:
            parts.append(self.subagents_index)

        if role != "triage":
            parts.append(
                "When the task is complete, reply with a final message and do "
                "not request any more tool calls."
            )

        return "\n\n".join(parts)


def load_domain(name: str) -> DomainConfig:
    """Load and cache a domain config from ``src/config/domains/{name}.yml``."""
    if name in _REGISTRY:
        return _REGISTRY[name]

    path = _DOMAINS_DIR / f"{name}.yml"
    if not path.exists():
        raise FileNotFoundError(
            f"Domain definition not found: {path}. "
            f"Available: {[p.stem for p in _DOMAINS_DIR.glob('*.yml')]}"
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    _validate_domain(raw, path)
    config = DomainConfig(_raw=raw)
    _REGISTRY[name] = config
    return config


def get_domain(name: str | None = None) -> DomainConfig:
    """Get a domain by name, defaulting to ``offsec``."""
    return load_domain(name or "offsec")


def _validate_domain(raw: dict, path: Path) -> None:
    """Basic structural validation of a domain YAML."""
    required_top = ["domain", "persona"]
    for key in required_top:
        if key not in raw:
            raise ValueError(f"Domain file {path} missing required key: {key!r}")

    domain = raw["domain"]
    if "name" not in domain:
        raise ValueError(f"Domain file {path} missing domain.name")

    persona = raw["persona"]
    for field_name in ("agent_name", "identity", "identity_short"):
        if field_name not in persona:
            raise ValueError(
                f"Domain file {path} missing persona.{field_name}"
            )
