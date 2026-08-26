"""Prefix Trie data structure for token-level constrained decoding.

Ensures that next-token candidates strictly follow valid token sequences.
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.models import TrieNode


class PrefixTrie(BaseModel):
    """Prefix Trie of token IDs to constrain generation."""

    model_config = ConfigDict(extra="ignore")

    root: TrieNode = Field(default_factory=TrieNode)

    def insert(self, token_ids: List[int], name: str) -> None:
        """Insert a token ID sequence into the prefix trie.

        Args:
            token_ids: Sequence of token IDs representing candidate text.
            name: The target name associated with this complete sequence.
        """
        current = self.root
        for token_id in token_ids:
            if token_id not in current.children:
                current.children[token_id] = TrieNode(token_id=token_id)
            current = current.children[token_id]
        current.is_terminal = True
        current.name = name

    def get_valid_next_tokens(self, node: TrieNode) -> List[int]:
        """Retrieve all valid next token IDs from a given node.

        Args:
            node: The current node in the trie.

        Returns:
            List of allowed next token IDs.
        """
        return list(node.children.keys())

    def advance(self, node: TrieNode, token_id: int) -> Optional[TrieNode]:
        """Advance to child node matching the token ID.

        Args:
            node: The current node in the trie.
            token_id: The chosen token ID.

        Returns:
            The child TrieNode if present, else None.
        """
        return node.children.get(token_id)
