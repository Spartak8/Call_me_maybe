"""Script for inspecting model vocabulary structure."""

import json
from llm_sdk import Small_LLM_Model


def main() -> None:
    """Load and print model vocabulary sample."""
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()

    with open(vocab_path, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    print("Vocabulary type:", type(vocab))
    print("Vocabulary size:", len(vocab))

    count = 0
    for token, token_id in vocab.items():
        print(repr(token), "->", token_id)
        count += 1
        if count == 20:
            break


if __name__ == "__main__":
    main()
