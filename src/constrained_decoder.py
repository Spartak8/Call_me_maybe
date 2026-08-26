"""Constrained decoding engine for function selection and parameter extraction.

Enforces schema compliance and structural JSON validity token-by-token.
"""

from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    PromptInput,
    TrieNode,
)
from src.trie import PrefixTrie


class ConstrainedDecoder(BaseModel):
    """Engine for performing constrained decoding with Small_LLM_Model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _build_system_prompt(
        self,
        functions: List[FunctionDefinition],
    ) -> str:
        """Construct the system prompt detailing available functions.

        Args:
            functions: List of available function definitions.

        Returns:
            Formatted system prompt string.
        """
        doc_lines: List[str] = []
        for fn in functions:
            param_types = ", ".join(
                f"{k}: {v.type}" for k, v in fn.parameters.items()
            )
            doc_lines.append(f"- {fn.name}({param_types}): {fn.description}")
        docs = "\n".join(doc_lines)
        return (
            "You are an expert function calling assistant.\n"
            "Given a user request, select the best function and "
            "provide the exact arguments.\n\n"
            f"Available functions:\n{docs}\n"
        )

    def _select_function(
        self,
        model: Small_LLM_Model,
        functions: List[FunctionDefinition],
        input_ids: List[int],
    ) -> Tuple[FunctionDefinition, List[int]]:
        """Select the target function via prefix trie constrained decoding.

        Args:
            model: The Small_LLM_Model instance.
            functions: List of available candidate functions.
            input_ids: Current sequence of token IDs including prompt prefix.

        Returns:
            Tuple containing the selected FunctionDefinition and updated
            token IDs.
        """
        trie = PrefixTrie()
        fn_map: Dict[str, FunctionDefinition] = {}

        for fn in functions:
            fn_map[fn.name] = fn
            encoded_tensor = model.encode(fn.name)
            fn_token_ids: List[int] = encoded_tensor[0].tolist()
            trie.insert(fn_token_ids, fn.name)

        current_node: TrieNode = trie.root
        curr_input_ids = list(input_ids)

        while True:
            if current_node.is_terminal and not current_node.children:
                break
            valid_next = trie.get_valid_next_tokens(current_node)
            if not valid_next:
                break
            if len(valid_next) == 1:
                chosen_token = valid_next[0]
            else:
                logits = model.get_logits_from_input_ids(curr_input_ids)
                chosen_token = max(
                    valid_next, key=lambda tid: logits[tid]
                )

            curr_input_ids.append(chosen_token)
            child = trie.advance(current_node, chosen_token)
            if child is None:
                break
            current_node = child

        if current_node.name and current_node.name in fn_map:
            return fn_map[current_node.name], curr_input_ids

        return functions[0], curr_input_ids

    def _decode_string_parameter(
        self,
        model: Small_LLM_Model,
        input_ids: List[int],
        max_tokens: int = 100,
    ) -> Tuple[str, List[int]]:
        """Decode a string parameter until closing quote.

        Args:
            model: The Small_LLM_Model instance.
            input_ids: Current input token ID sequence.
            max_tokens: Maximum tokens allowed for the string value.

        Returns:
            Tuple of extracted string value and updated token ID sequence.
        """
        gen_ids: List[int] = []
        curr_ids = list(input_ids)

        for _ in range(max_tokens):
            logits = model.get_logits_from_input_ids(curr_ids)
            best_id = max(range(len(logits)), key=lambda i: logits[i])
            decoded_char = model.decode([best_id])
            if '"' in decoded_char:
                parts = decoded_char.split('"', 1)
                if parts[0]:
                    curr_ids.append(best_id)
                    gen_ids.append(best_id)
                break
            curr_ids.append(best_id)
            gen_ids.append(best_id)

        raw_str = model.decode(gen_ids)
        if '"' in raw_str:
            raw_str = raw_str.split('"', 1)[0]
        return raw_str.strip(), curr_ids

    def _decode_number_parameter(
        self,
        model: Small_LLM_Model,
        input_ids: List[int],
        max_tokens: int = 25,
    ) -> Tuple[float, List[int]]:
        """Decode a numerical parameter until delimiter.

        Args:
            model: The Small_LLM_Model instance.
            input_ids: Current input token ID sequence.
            max_tokens: Maximum tokens allowed for the numerical value.

        Returns:
            Tuple of extracted float value and updated token ID sequence.
        """
        gen_ids: List[int] = []
        curr_ids = list(input_ids)

        for _ in range(max_tokens):
            logits = model.get_logits_from_input_ids(curr_ids)
            best_id = max(range(len(logits)), key=lambda i: logits[i])
            dec = model.decode([best_id])
            if any(c in dec for c in [",", "}", " ", "\n", '"', "]"]):
                break
            curr_ids.append(best_id)
            gen_ids.append(best_id)

        raw_num = model.decode(gen_ids).strip()
        try:
            val = float(raw_num)
        except ValueError:
            val = 0.0
        return val, curr_ids

    def _decode_boolean_parameter(
        self,
        model: Small_LLM_Model,
        input_ids: List[int],
    ) -> Tuple[bool, List[int]]:
        """Decode a boolean parameter by comparing true/false logits.

        Args:
            model: The Small_LLM_Model instance.
            input_ids: Current input token ID sequence.

        Returns:
            Tuple of extracted boolean value and updated token ID sequence.
        """
        logits = model.get_logits_from_input_ids(input_ids)
        true_ids: List[int] = model.encode("true")[0].tolist()
        false_ids: List[int] = model.encode("false")[0].tolist()

        score_true = logits[true_ids[0]]
        score_false = logits[false_ids[0]]

        curr_ids = list(input_ids)
        if score_true >= score_false:
            curr_ids.extend(true_ids)
            return True, curr_ids

        curr_ids.extend(false_ids)
        return False, curr_ids

    def process_prompt(
        self,
        model: Small_LLM_Model,
        prompt_input: PromptInput,
        functions: List[FunctionDefinition],
    ) -> FunctionCallResult:
        """Process a single prompt and produce the validated function call.

        Args:
            model: The Small_LLM_Model instance.
            prompt_input: The user's input prompt model.
            functions: List of available function definitions.

        Returns:
            Validated FunctionCallResult model.
        """
        sys_msg = self._build_system_prompt(functions)
        user_prompt = prompt_input.prompt

        full_prompt = (
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
            '{"name": "'
        )

        input_ids: List[int] = model.encode(full_prompt)[0].tolist()

        chosen_fn, input_ids = self._select_function(
            model=model,
            functions=functions,
            input_ids=input_ids,
        )

        prefix_params = '", "parameters": {'
        input_ids.extend(model.encode(prefix_params)[0].tolist())

        param_items = list(chosen_fn.parameters.items())
        param_values: Dict[str, Any] = {}

        for idx, (p_name, p_info) in enumerate(param_items):
            p_prefix = f'"{p_name}": '
            if p_info.type == "string":
                p_prefix += '"'
            input_ids.extend(model.encode(p_prefix)[0].tolist())

            if p_info.type == "string":
                str_val, input_ids = self._decode_string_parameter(
                    model, input_ids
                )
                param_values[p_name] = str_val
            elif p_info.type in ["number", "float", "integer", "int"]:
                num_val, input_ids = self._decode_number_parameter(
                    model, input_ids
                )
                param_values[p_name] = num_val
            elif p_info.type == "boolean":
                bool_val, input_ids = self._decode_boolean_parameter(
                    model, input_ids
                )
                param_values[p_name] = bool_val
            else:
                str_val, input_ids = self._decode_string_parameter(
                    model, input_ids
                )
                param_values[p_name] = str_val

            if idx < len(param_items) - 1:
                input_ids.extend(model.encode(", ")[0].tolist())
            else:
                input_ids.extend(model.encode("}")[0].tolist())

        return FunctionCallResult(
            prompt=user_prompt,
            name=chosen_fn.name,
            parameters=param_values,
        )
