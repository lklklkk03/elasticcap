"""
ChainCaps: Declassification Token Issuer (unchanged from ChainCaps)

Trusted component that issues HMAC-signed declassification tokens.
This runs in the trusted issuer context (system/user), NOT in the LLM.
"""

import hashlib
import hmac
import json
import uuid
from typing import List

from .budget import SinkPrivilege


def issue_declassification_token(
    signing_key: bytes,
    sink_privilege: SinkPrivilege,
    lineage_node_ids: List[str],
) -> str:
    """Issue a cryptographically signed declassification token."""
    payload = json.dumps({
        "token_id": f"tok_{uuid.uuid4().hex[:12]}",
        "sink": str(sink_privilege),
        "lineage": lineage_node_ids,
        "user_approval": True,
        "scope": "one-shot",
    }, sort_keys=True)

    token_hmac = hmac.new(
        signing_key, payload.encode(), hashlib.sha256
    ).hexdigest()

    return json.dumps({
        "payload": payload,
        "hmac": token_hmac,
    })
