"""
ElasticCap: Elastic Capability Control for LLM Agent Composition Safety

A drop-in improvement over ChainCaps that adds:
1. Dependency-chain-aware scope isolation (vs global context budget)
2. Automatic budget release when dependency chains complete
3. Declassification auditor for sanitize-then-share workflows

Based on ChainCaps (Jiang et al., ICML 2026 AIWILD Workshop).
"""

from .budget import (
    Budget,
    SinkPrivilege,
    SinkType,
    budget_display_only,
    budget_internal_email,
    budget_public,
    budget_sensitive_file,
)
from .dag import ElasticDAG, DataNode, SinkEvent
from .manifest import ToolManifest, get_manifest, STANDARD_MANIFESTS
from .token_issuer import issue_declassification_token
from .auditor import DeclassificationAuditor, AuditResult
from .engine import ElasticCapEngine, ToolCall, ToolResult, EnforcementStats
