"""Methods available from the command line. Add your own by subclassing
`Method` and registering it here (see docs/methods.md)."""

from .base import EpisodeResult, Method
from .baselines import Blind, Oracle
from .rag import RAG

METHODS = {"blind": Blind, "oracle": Oracle, "rag": RAG}

__all__ = ["METHODS", "Method", "EpisodeResult", "Blind", "Oracle", "RAG"]
