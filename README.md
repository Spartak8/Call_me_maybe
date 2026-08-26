*This project has been created as part of the 42 curriculum by skhachat.*

# Call Me Maybe: Function Calling in LLMs

## Description
**Call Me Maybe** is a high-reliability function calling engine built for lightweight Large Language Models (specifically `Qwen/Qwen3-0.6B`). Small language models (under 1 billion parameters) typically struggle to output syntactically valid JSON and conform strictly to function calling schemas when prompted with raw natural language.

This project solves that limitation through **Constrained Decoding**—a technique that guides the language model token-by-token using structural prefix trees (Tries) and logit masking. By restricting candidate tokens at generation time to only those that maintain syntactic and schema validity, the engine guarantees 100% parseable JSON output, exact type compliance, and deterministic schema adherence.

---

## Instructions

### Prerequisites
- Python >= 3.10
- `uv` package manager

### Installation
To install all project dependencies into the virtual environment:
```bash
make install
# or
uv sync
```

### Execution
Run the default pipeline (processes `data/input/function_calling_tests.json` using definitions from `data/input/functions_definition.json` and outputs to `data/output/function_calling_results.json`):
```bash
make run
# or
uv run python -m src
```

You can also specify custom input, output, or model paths:
```bash
uv run python -m src \
  --functions_definition <path_to_definitions> \
  --input <path_to_input_prompts> \
  --output <path_to_output_results> \
  --model_name <hf_model_id>
```

### Debugging
To execute the pipeline under Python's built-in debugger (`pdb`):
```bash
make debug
```

### Linting and Type Checking
Run strict flake8 and mypy checks:
```bash
make lint
# or for strict mode:
make lint-strict
```

### Testing
Run the automated unit and integration test suite:
```bash
uv run python -m unittest discover tests
```

### Cleaning
Remove temporary caches, bytecodes, and generated outputs:
```bash
make clean
```

---

## Algorithm Explanation

### The Generation Pipeline
1. **Prompt Formatting**: System instructions and schema definitions of candidate functions are formatted into the model's native ChatML template (`<|im_start|>system...<|im_end|>\n<|im_start|>user...<|im_end|>\n<|im_start|>assistant\n{"name": "`).
2. **Function Name Selection via Prefix Trie**:
   - Every available function name from `functions_definition.json` is encoded into its constituent token IDs.
   - A **Prefix Trie** (`PrefixTrie`) is constructed from these token sequences.
   - At each generation step, the engine inspects the current Trie node to identify all valid continuation token IDs.
   - The LLM's next-token logits are computed via `model.get_logits_from_input_ids(input_ids)`.
   - Logits of invalid tokens are masked, and the valid token with the highest logit score is selected.
   - The engine traverses down the Trie until a terminal leaf node is reached, selecting the function deterministically according to the LLM's probability distribution.
3. **Structured Parameter Decoding**:
   - The assistant prompt is extended with `", "parameters": {"`.
   - For each parameter declared in the function's schema:
     - **String parameters**: Generated token-by-token until a closing quotation mark `"` is encountered or length limit is reached.
     - **Numerical parameters**: Decoded until delimiter boundaries (`,`, `}`, space, newline) and safely cast to `float`/`int`.
     - **Boolean parameters**: Evaluated by comparing the model's logits for `true` vs `false` token continuations.
   - JSON delimiters (`,` between arguments and closing `}`) are inserted structurally.

---

## Bonus Features Implemented

1. **Multiple LLM Model Support**: Configurable via `--model_name <model_id>` CLI parameter, supporting alternative models (such as `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`, etc.).
2. **Recoded Custom Tokenizer (`src/tokenizer.py`)**: Implements token-level mapping and decoding methods directly from `vocab.json` using `get_path_to_vocab_file()`.
3. **Advanced Error Recovery & Fallbacks**: Comprehensive error protection in IO handling and decoding pipeline ensuring non-crashing behavior under corrupted inputs or missing schema fields.
4. **Real-time Terminal Output & Progress Visualization**: Displays per-prompt function resolution and parameter extraction in real time.
5. **Comprehensive Test Suite**: Automated unit and integration tests covering Pydantic models, prefix tries, tokenizer, and IO operations.

---

## Design Decisions
- **100% Pydantic Model Architecture**: In accordance with project requirements, all data models (`FunctionParameter`, `FunctionDefinition`, `PromptInput`, `FunctionCallResult`, `TrieNode`, `PrefixTrie`, `CustomTokenizer`, `ConstrainedDecoder`, `IOHandler`) inherit from Pydantic `BaseModel` for validation, schema enforcement, and serialization.
- **Pure Logits-Driven Decision Making**: Function selection is guided entirely by LLM token logits without regex heuristics or hardcoded keywords.
- **Zero Heavy External Dependencies in Source**: No forbidden packages (`dspy`, `transformers`, `torch`, `outlines`) are directly imported in `src/`. The engine interacts exclusively through the public API of `Small_LLM_Model`.
- **Fault-Tolerant I/O**: File operations utilize context managers, and all missing file or corrupted JSON scenarios produce descriptive error messages without crashing.

---

## Performance Analysis
- **Accuracy**: Achieves near-perfect function selection accuracy on benchmark test sets by constraining the search space to valid function names.
- **JSON Validity**: 100% guaranteed syntactic validity because structural JSON tokens (`{`, `}`, `"`, `: `, `, `) are controlled directly by the decoder state machine.
- **Execution Speed**: Fully processes the benchmark test cases in approximately 1 to 2 minutes on standard CPU hardware.

---

## Challenges Faced & Solutions
1. **Model Thinking/Reasoning Prefix**: Small reasoning models often prepend `<think>...</think>` tokens when given unstructured prompts.
   - *Solution*: Prefilling the assistant response with `{"name": "` immediately bypasses the chain-of-thought preamble and forces the model into structured generation mode.
2. **Subword Tokenization Quirks**: Tokenizers split identifiers (such as `fn_add_numbers`) into multiple subwords (`fn`, `_add`, `_numbers`).
   - *Solution*: Implementing a multi-level token Prefix Trie ensures multi-token names are navigated step-by-step without dropping intermediate tokens.
3. **Leading Space Artifacts**: BPE tokenizers frequently produce space-prefixed tokens (e.g. `' *'`).
   - *Solution*: Parameter extraction routines clean boundary delimiters while preserving inner whitespace.

---

## Testing Strategy
- **Unit Tests (`tests/test_constrained_decoder.py`)**:
  - Pydantic schema validation tests (valid vs missing fields, forbidden extra attributes).
  - Prefix Trie insertion, token branch matching, and terminal detection tests.
  - Custom Tokenizer vocabulary decoding tests.
  - I/O handler tests verifying graceful handling of missing files, malformed JSON, and invalid schemas.
- **Integration Testing**:
  - Full end-to-end execution against `data/input/functions_definition.json` and `data/input/function_calling_tests.json`.
  - Static type checking with `mypy --strict` and style verification with `flake8`.

---

## Example Usage

### 1. Default Run
```bash
uv run python -m src
```

### 2. Custom Paths & Model
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/my_results.json \
  --model_name Qwen/Qwen3-0.6B
```

---

## Resources & AI Usage
- **References**:
  - *Outlines & Grammar-Based Decoding*: Structured generation in Language Models via finite-state machines.
  - *Qwen Technical Reports*: Architecture and tokenization of the Qwen model family.
  - *Hugging Face Tokenizers Documentation*: BPE tokenization and vocabulary representations.
  - *Pydantic Documentation*: Data validation using Python type annotations.
- **AI Usage**:
  - AI was utilized as an assistive pair-programmer to explore token-level Trie generation strategies, write test assertions, and format documentation in accordance with PEP 257 and flake8 standards.
