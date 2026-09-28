from __future__ import annotations

from typing import Any, Dict, Optional

from ....engine.tool import Tool
from .. import docs as docs_mod


class ReadDocs(Tool):
    """Section-level retrieval over the store's reference documents.

    read_docs(doc)           -> table of contents (section slugs)
    read_docs(doc, section)  -> that section's content (with any amendments merged)

    This base class serves the base documents; each tier's docs module defines
    its own reader over the tier's frozen documents.
    """

    STAGE = 0

    @classmethod
    def invoke(cls, data: Dict[str, Any], doc: str, section: Optional[str] = None) -> str:  # type: ignore[override]
        return docs_mod.lookup(doc, cls.STAGE, section)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "read_docs",
                "description": (
                    "Read the store's reference documents at SECTION granularity. Call with only `doc` "
                    "to get the table of contents (section slugs); call again with `section` to read that "
                    "section. 'policy' holds the rules (what may / may not be done); 'workflow' holds "
                    "operating procedures and domain reference. Consult the relevant policy section "
                    "before any write action or refusal."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc": {"type": "string", "enum": ["policy", "workflow"],
                                "description": "Which document."},
                        "section": {"type": "string",
                                    "description": "Section slug from the table of contents, e.g. 'cancellation'. Omit to list sections."},
                    },
                    "required": ["doc"],
                },
            },
        }
