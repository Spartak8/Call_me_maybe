"""Entry point for the Function Calling CLI application.

Processes input prompts and generates schema-compliant function calls.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from llm_sdk import Small_LLM_Model
from src.constrained_decoder import ConstrainedDecoder
from src.io_handler import IOHandler
from src.models import FunctionCallResult


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path to write the resulting JSON array.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Name or path of the LLM model to load.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the main function calling pipeline."""
    args = parse_arguments()
    io_handler = IOHandler()

    functions = io_handler.load_functions(args.functions_definition)
    if functions is None:
        sys.exit(1)

    prompts = io_handler.load_prompts(args.input)
    if prompts is None:
        sys.exit(1)

    print(f"Loaded {len(functions)} functions and {len(prompts)} prompts.\n")

    try:
        model = Small_LLM_Model(model_name=args.model_name)
    except Exception as err:
        print(f"Error: Failed to initialize LLM model: {err}", file=sys.stderr)
        sys.exit(1)

    decoder = ConstrainedDecoder()
    results: List[FunctionCallResult] = []

    for idx, prompt_input in enumerate(prompts, start=1):
        print(f"[{idx}/{len(prompts)}] Prompt: {prompt_input.prompt}")
        try:
            res = decoder.process_prompt(
                model=model,
                prompt_input=prompt_input,
                functions=functions,
            )
            results.append(res)
            print(f" -> Function  : {res.name}")
            print(f" -> Parameters: {json.dumps(res.parameters)}")
            print()
        except Exception as err:
            print(
                f"Error processing prompt '{prompt_input.prompt}': {err}",
                file=sys.stderr,
            )
            fallback_res = FunctionCallResult(
                prompt=prompt_input.prompt,
                name=functions[0].name,
                parameters={},
            )
            results.append(fallback_res)
            print(f" -> Fallback Function: {fallback_res.name}\n")

    saved = io_handler.save_results(args.output, results)
    if not saved:
        sys.exit(1)

    print(
        f"Successfully generated {len(results)} function call results "
        f"to '{args.output}'"
    )


if __name__ == "__main__":
    main()
