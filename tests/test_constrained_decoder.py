"""Unit and integration tests for function calling and constrained decoding."""

import json
import tempfile
import unittest
from pathlib import Path
from pydantic import ValidationError

from src.io_handler import IOHandler
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    FunctionParameter,
    PromptInput,
)
from src.tokenizer import CustomTokenizer
from src.trie import PrefixTrie


class TestModels(unittest.TestCase):
    """Test Pydantic model validation."""

    def test_function_parameter_valid(self) -> None:
        param = FunctionParameter(type="string", description="A string")
        self.assertEqual(param.type, "string")
        self.assertEqual(param.description, "A string")

    def test_function_definition_valid(self) -> None:
        fn = FunctionDefinition(
            name="fn_test",
            description="Test function",
            parameters={"a": FunctionParameter(type="number")},
        )
        self.assertEqual(fn.name, "fn_test")
        self.assertIn("a", fn.parameters)

    def test_prompt_input_valid(self) -> None:
        inp = PromptInput(prompt="Hello world")
        self.assertEqual(inp.prompt, "Hello world")

    def test_function_call_result_valid(self) -> None:
        res = FunctionCallResult(
            prompt="Test",
            name="fn_test",
            parameters={"a": 42.0},
        )
        self.assertEqual(res.name, "fn_test")
        self.assertEqual(res.parameters["a"], 42.0)

    def test_function_call_result_extra_forbidden(self) -> None:
        with self.assertRaises(ValidationError):
            FunctionCallResult.model_validate(
                {"prompt": "x", "name": "fn", "parameters": {}, "extra": 1}
            )


class TestPrefixTrie(unittest.TestCase):
    """Test token prefix trie behavior."""

    def test_trie_insertion_and_traversal(self) -> None:
        trie = PrefixTrie()
        trie.insert([10, 20, 30], "fn_a")
        trie.insert([10, 20, 40], "fn_b")
        trie.insert([15, 25], "fn_c")

        root = trie.root
        valid_next = trie.get_valid_next_tokens(root)
        self.assertCountEqual(valid_next, [10, 15])

        child_10 = trie.advance(root, 10)
        self.assertIsNotNone(child_10)
        assert child_10 is not None
        valid_next_10 = trie.get_valid_next_tokens(child_10)
        self.assertEqual(valid_next_10, [20])

        child_20 = trie.advance(child_10, 20)
        self.assertIsNotNone(child_20)
        assert child_20 is not None
        valid_next_20 = trie.get_valid_next_tokens(child_20)
        self.assertCountEqual(valid_next_20, [30, 40])

        child_30 = trie.advance(child_20, 30)
        self.assertIsNotNone(child_30)
        assert child_30 is not None
        self.assertTrue(child_30.is_terminal)
        self.assertEqual(child_30.name, "fn_a")


class TestIOHandler(unittest.TestCase):
    """Test input/output handling and error resilience."""

    def setUp(self) -> None:
        self.handler = IOHandler()

    def test_load_nonexistent_file(self) -> None:
        res = self.handler.load_functions(Path("/nonexistent/path.json"))
        self.assertIsNone(res)

    def test_load_invalid_json(self) -> None:
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".json"
        ) as tmp:
            tmp.write("{ invalid json }")
            tmp_path = Path(tmp.name)
        try:
            res = self.handler.load_functions(tmp_path)
            self.assertIsNone(res)
        finally:
            tmp_path.unlink()

    def test_load_invalid_schema(self) -> None:
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".json"
        ) as tmp:
            tmp.write(json.dumps([{"invalid_field": 123}]))
            tmp_path = Path(tmp.name)
        try:
            res = self.handler.load_functions(tmp_path)
            self.assertIsNone(res)
        finally:
            tmp_path.unlink()

    def test_save_and_load_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "output" / "results.json"
            results = [
                FunctionCallResult(
                    prompt="Calculate 1+1",
                    name="fn_add",
                    parameters={"a": 1.0, "b": 1.0},
                )
            ]
            saved = self.handler.save_results(out_path, results)
            self.assertTrue(saved)
            self.assertTrue(out_path.exists())

            with out_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["name"], "fn_add")


class TestCustomTokenizer(unittest.TestCase):
    """Test custom tokenizer implementation."""

    def test_custom_tokenizer_decode(self) -> None:
        tok = CustomTokenizer(
            vocab={"Hello": 0, "Ġworld": 1},
            inverse_vocab={0: "Hello", 1: "Ġworld"},
        )
        self.assertEqual(tok.decode_token_id(0), "Hello")
        self.assertEqual(tok.decode_token_ids([0, 1]), "Hello world")


if __name__ == "__main__":
    unittest.main()
