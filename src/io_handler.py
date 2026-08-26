"""Input/output handling for function calling tests and schemas.

Provides resilient JSON loading and writing with comprehensive error checking.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, ValidationError

from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    PromptInput,
)


class IOHandler(BaseModel):
    """Handler for loading inputs, validating schemas, and writing outputs."""

    model_config = ConfigDict(extra="ignore")

    def load_json_array(self, path: Path) -> Optional[List[Dict[str, Any]]]:
        """Safely load a JSON array from the given path.

        Args:
            path: Path to the target JSON file.

        Returns:
            Parsed list of dictionaries, or None if reading/parsing fails.
        """
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError:
            print(f"Error: File not found at '{path}'", file=sys.stderr)
            return None
        except json.JSONDecodeError as err:
            print(
                f"Error: Malformed JSON in '{path}': {err}",
                file=sys.stderr,
            )
            return None
        except OSError as err:
            print(
                f"Error: Cannot access file '{path}': {err}",
                file=sys.stderr,
            )
            return None

        if not isinstance(data, list):
            print(
                f"Error: Expected JSON array in '{path}', got {type(data)}",
                file=sys.stderr,
            )
            return None

        return data

    def load_functions(
        self,
        path: Path,
    ) -> Optional[List[FunctionDefinition]]:
        """Load and validate function definitions.

        Args:
            path: Path to the function definitions JSON file.

        Returns:
            List of validated FunctionDefinition models, or None on failure.
        """
        data = self.load_json_array(path)
        if data is None:
            return None

        functions: List[FunctionDefinition] = []
        for idx, item in enumerate(data):
            try:
                functions.append(FunctionDefinition.model_validate(item))
            except ValidationError as err:
                print(
                    f"Error: Invalid function schema at index {idx} in "
                    f"'{path}': {err}",
                    file=sys.stderr,
                )
                return None

        if not functions:
            print(
                f"Error: No valid functions found in '{path}'",
                file=sys.stderr,
            )
            return None

        return functions

    def load_prompts(self, path: Path) -> Optional[List[PromptInput]]:
        """Load and validate test prompt inputs.

        Args:
            path: Path to the prompt inputs JSON file.

        Returns:
            List of validated PromptInput models, or None on failure.
        """
        data = self.load_json_array(path)
        if data is None:
            return None

        prompts: List[PromptInput] = []
        for idx, item in enumerate(data):
            try:
                prompts.append(PromptInput.model_validate(item))
            except ValidationError as err:
                print(
                    f"Error: Invalid prompt schema at index {idx} in "
                    f"'{path}': {err}",
                    file=sys.stderr,
                )
                return None

        return prompts

    def save_results(
        self,
        path: Path,
        results: List[FunctionCallResult],
    ) -> bool:
        """Write function call results to the specified output JSON file.

        Args:
            path: Destination file path.
            results: List of FunctionCallResult models to write.

        Returns:
            True if saving succeeded, False otherwise.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            output_data = [res.model_dump() for res in results]
            with path.open("w", encoding="utf-8") as file:
                json.dump(output_data, file, indent=2)
            return True
        except OSError as err:
            print(
                f"Error: Failed to write output file '{path}': {err}",
                file=sys.stderr,
            )
            return False
