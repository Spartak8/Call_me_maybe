"""Script for testing Small_LLM_Model encoding and logits retrieval."""

from llm_sdk import Small_LLM_Model


def main() -> None:
    """Execute basic LLM model inference and vocabulary checks."""
    model = Small_LLM_Model()

    text = "Hello"
    ids = model.encode(text)

    print("Text:", text)
    print("Token IDs:", ids)
    print("Decoded:", model.decode(ids[0]))

    input_ids = ids[0].tolist()
    logits = model.get_logits_from_input_ids(input_ids)

    print("Number of logits:", len(logits))
    print("First 5 logits:", logits[:5])

    vocab_path = model.get_path_to_vocab_file()
    print("Vocabulary:", vocab_path)


if __name__ == "__main__":
    main()
