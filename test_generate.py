"""Script for testing single token generation step."""

from llm_sdk import Small_LLM_Model


def main() -> None:
    """Predict single next token from input prompt."""
    model = Small_LLM_Model()

    prompt = "The capital of France is"
    encoded = model.encode(prompt)
    input_ids = encoded[0].tolist()

    logits = model.get_logits_from_input_ids(input_ids)

    next_token_id = max(
        range(len(logits)),
        key=lambda index: logits[index],
    )

    next_token = model.decode([next_token_id])

    print("Prompt:", prompt)
    print("Next token ID:", next_token_id)
    print("Next token:", repr(next_token))


if __name__ == "__main__":
    main()
