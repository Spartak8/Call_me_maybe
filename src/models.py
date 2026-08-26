"""Data models for function calling and constrained decoding.

All data structures are defined as Pydantic models to ensure strict validation.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class FunctionParameter(BaseModel):
    """Schema definition for an individual function parameter."""

    model_config = ConfigDict(extra="ignore")

    type: str
    description: Optional[str] = None


class FunctionDefinition(BaseModel):
    """Schema definition for an available tool/function."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    parameters: Dict[str, FunctionParameter] = Field(default_factory=dict)
    returns: Optional[FunctionParameter] = None


class PromptInput(BaseModel):
    """Input prompt representation."""

    model_config = ConfigDict(extra="ignore")

    prompt: str


class FunctionCallResult(BaseModel):
    """Structured output representation for a resolved function call."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: Dict[str, Any]


class TrieNode(BaseModel):
    """Node in the prefix trie representing a token in a sequence."""

    model_config = ConfigDict(extra="ignore")

    token_id: Optional[int] = None
    children: Dict[int, "TrieNode"] = Field(default_factory=dict)
    is_terminal: bool = False
    name: Optional[str] = None


# Enable recursive model definition for TrieNode
TrieNode.model_rebuild()
