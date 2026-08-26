"""Custom Tokenizer implementation for bonus requirements.

Implements tokenization and decoding using vocabulary files directly.
"""

import json
from typing import Dict, List
from pydantic import BaseModel, ConfigDict, Field


class CustomTokenizer(BaseModel):
    """Custom Tokenizer built from model vocabulary file."""

    model_config = ConfigDict(extra="ignore")

    vocab: Dict[str, int] = Field(default_factory=dict)
    inverse_vocab: Dict[int, str] = Field(default_factory=dict)

    @classmethod
    def from_vocab_file(cls, vocab_path: str) -> "CustomTokenizer":
        """Load tokenizer vocabulary from the given vocab JSON file.

        Args:
            vocab_path: Path string to the vocab.json file.

        Returns:
            Instantiated CustomTokenizer model.
        """
        with open(vocab_path, "r", encoding="utf-8") as file:
            vocab_data = json.load(file)
        inv_vocab = {int(v): str(k) for k, v in vocab_data.items()}
        return cls(vocab=vocab_data, inverse_vocab=inv_vocab)

    def decode_token_id(self, token_id: int) -> str:
        """Decode a single token ID to its string representation.

        Args:
            token_id: Integer ID of the token.

        Returns:
            Decoded string representation.
        """
        return self.inverse_vocab.get(token_id, "")

    def decode_token_ids(self, token_ids: List[int]) -> str:
        """Decode a sequence of token IDs to text string.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded string with byte/space representations normalized.
        """
        raw_text = "".join(self.decode_token_id(tid) for tid in token_ids)
        return raw_text.replace("Ġ", " ").replace("Ċ", "\n")
