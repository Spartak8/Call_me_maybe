"""Script for testing multi-step token generation loop."""

from llm_sdk import Small_LLM_Model


def main() -> None:
    """Generate a sequence of tokens autoregressively."""
    model = Small_LLM_Model()

    prompt = "The capital of France is"
    encoded = model.encode(prompt)
    input_ids = encoded[0].tolist()

    generated_ids = []

    for _ in range(10):
        logits = model.get_logits_from_input_ids(input_ids)

        next_token_id = max(
            range(len(logits)),
            key=lambda index: logits[index],
        )

        input_ids.append(next_token_id)
        generated_ids.append(next_token_id)

    generated_text = model.decode(generated_ids)

    print("Prompt:", prompt)
    print("Generated:", repr(generated_text))


if __name__ == "__main__":
    main()
