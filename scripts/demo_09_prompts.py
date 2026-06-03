"""Demo for Component 9 (prompts).

Run:
    conda activate medium-rag
    python scripts/demo_09_prompts.py

Renders the fixed System prompt and a User prompt built from a few hand-written
fake chunks (one with empty authors to show the `Unknown` fallback), then the
empty-context path so the '(no relevant context retrieved)' marker is visible.

No Pinecone/LLM calls -- costs $0. No CLI args. ASCII-only output (this console
is cp1252). Exits non-zero only on an unexpected exception (e.g. the
original-question-present assertion fails).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompts import SYSTEM_PROMPT, build_augmented_prompt, render_user_prompt


@dataclass
class _FakeChunk:
    article_id: str
    title: str
    authors: list = field(default_factory=list)
    chunk: str = ""


def main() -> int:
    print("=== SYSTEM PROMPT ===")
    print(SYSTEM_PROMPT)

    chunks = [
        _FakeChunk(
            "42",
            "Marketing as Conversation",
            ["Jane Doe"],
            "Treat promotion as a dialogue ...",
        ),
        _FakeChunk(
            "117",
            "Habits That Stick",
            [],  # -> Unknown
            "Start absurdly small ...",
        ),
    ]
    q = "Which article would you recommend for building habits that stick, and why?"

    print("\n=== USER PROMPT ===")
    user = render_user_prompt(q, chunks)
    print(user)
    assert q in user, "original question must appear verbatim"

    print("\n=== USER PROMPT (no context) ===")
    print(render_user_prompt(q, []))

    # Show the capitalized-key contract C11/C12 depend on.
    ap = build_augmented_prompt(q, chunks)
    assert set(ap.keys()) == {"System", "User"}, "Augmented_prompt keys must be System/User"

    print(
        "\nOK: system + user prompts rendered; original question present; "
        "empty-context marker shown."
    )
    return 0


raise SystemExit(main())
