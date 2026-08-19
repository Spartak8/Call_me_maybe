import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError


class FunctionParameter(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, FunctionParameter]
    returns: FunctionParameter


class PromptInput(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, object]


def load_json(path: Path) -> list[dict[str, object]] | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: invalid JSON: {path}")
        return None
    except OSError as error:
        print(f"Error: cannot read {path}: {error}")
        return None

    if not isinstance(data, list):
        print(f"Error: expected JSON array: {path}")
        return None

    return data


def validate_functions(
    data: list[dict[str, object]],
) -> list[FunctionDefinition] | None:
    try:
        return [
            FunctionDefinition.model_validate(item)
            for item in data
        ]
    except ValidationError as error:
        print(f"Error: invalid function definition: {error}")
        return None


def validate_prompts(
    data: list[dict[str, object]],
) -> list[PromptInput] | None:
    try:
        return [
            PromptInput.model_validate(item)
            for item in data
        ]
    except ValidationError as error:
        print(f"Error: invalid prompt: {error}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
    )
    args = parser.parse_args()

    raw_functions = load_json(args.functions_definition)
    raw_prompts = load_json(args.input)

    if raw_functions is None or raw_prompts is None:
        sys.exit(1)

    functions = validate_functions(raw_functions)
    prompts = validate_prompts(raw_prompts)

    if functions is None or prompts is None:
        sys.exit(1)

    print(f"Loaded {len(functions)} functions")
    print(f"Loaded {len(prompts)} prompts")


if __name__ == "__main__":
    main()
