# TOON Research & Implementation Plan

**Project**: ccflow (Claude Code CLI ↔ SDK Middleware)
**Date**: 2025-12-31
**Status**: Research Complete

---

## Executive Summary

**TOON (Token-Oriented Object Notation)** is a production-ready data serialization format designed specifically for LLM input optimization. It achieves **30-60% token reduction** compared to JSON while maintaining lossless round-trip compatibility.

**Key Finding**: An official Python implementation exists (`toon-python`) with 792 tests and 91% coverage. We should **integrate this library** rather than building our own.

---

## 1. TOON Format Specification

### 1.1 Core Design Principles

TOON combines:
- **YAML-style indentation** for nested objects (no braces)
- **CSV-style tabular layout** for uniform arrays (header + rows)
- **Explicit structure declarations** (`[N]` lengths, `{fields}` headers)

### 1.2 Format Examples

**JSON Input:**
```json
{
  "context": {
    "task": "Portfolio analysis",
    "account": "U1234567"
  },
  "positions": [
    {"symbol": "AAPL", "qty": 100, "avgCost": 150.25},
    {"symbol": "GOOGL", "qty": 50, "avgCost": 2800.00}
  ]
}
```

**TOON Output (39% fewer tokens):**
```
context:
  task: Portfolio analysis
  account: U1234567
positions[2]{symbol,qty,avgCost}:
  AAPL,100,150.25
  GOOGL,50,2800
```

### 1.3 Syntax Rules

| Element | Syntax | Example |
|---------|--------|---------|
| Object | `key: value` + indentation | `name: Alice` |
| Nested object | Indented children | `user:\n  id: 1` |
| Primitive array | `key[N]: v1,v2,v3` | `tags[3]: a,b,c` |
| Tabular array | `key[N]{f1,f2}: rows...` | `items[2]{id,name}:\n  1,Alice\n  2,Bob` |
| Mixed array | `key[N]:\n  - item` | Hyphen-prefixed list items |

### 1.4 Quoting Rules

Strings require quotes only when containing:
- Empty string (`""`)
- Leading/trailing whitespace
- Reserved words (`true`, `false`, `null`)
- Numeric patterns (`42`, `-3.14`)
- Special chars (`:`, `"`, `\`, `[`, `]`, `{`, `}`)
- Active delimiter (comma by default)
- Leading hyphen (`-`)

### 1.5 Escape Sequences

Only five valid escapes: `\\`, `\"`, `\n`, `\r`, `\t`

### 1.6 Number Canonicalization

- No exponent notation: `1e6` → `1000000`
- No leading zeros: `007` → `7`
- No trailing zeros: `3.140` → `3.14`
- Normalize `-0` → `0`

### 1.7 Type Normalization

| Python Type | TOON Output |
|-------------|-------------|
| `float('inf')` | `null` |
| `float('nan')` | `null` |
| `Decimal` | `float` |
| `datetime` | ISO 8601 string |
| `None` | `null` |

---

## 2. Performance Benchmarks

### 2.1 Token Efficiency

| Data Type | JSON Tokens | TOON Tokens | Savings |
|-----------|-------------|-------------|---------|
| Flat tabular (100% uniform) | 164,254 | 67,695 | **58.8%** |
| Mixed structure | 289,901 | 226,613 | **21.8%** |
| Semi-uniform (40-60%) | Variable | Variable | 10-30% |

### 2.2 Accuracy Benchmarks (209 retrieval tasks, 4 LLMs)

| Format | Accuracy | Tokens Used |
|--------|----------|-------------|
| **TOON** | 73.9% | 2,744 |
| JSON | 69.7% | 4,545 |

**Result**: TOON achieves **higher accuracy** with **39.6% fewer tokens**.

### 2.3 Per-Model Performance

| Model | TOON Accuracy | JSON Accuracy |
|-------|---------------|---------------|
| Claude Haiku | 59.8% | 57.4% |
| Gemini 2.5 Flash | 87.6% | 77.0% |
| GPT-5 Nano | 90.9% | 90.9% |

---

## 3. Official Python Implementation

### 3.1 Package Details

- **Repository**: https://github.com/toon-format/toon-python
- **Version**: 0.9.x (beta, targeting 1.0 spec compliance)
- **Python**: 3.8+ required
- **Tests**: 792 tests, 91% coverage
- **License**: MIT

### 3.2 Installation

```bash
# From GitHub (current)
pip install git+https://github.com/toon-format/toon-python.git

# Future PyPI (when available)
pip install toon-format
```

### 3.3 Core API

```python
from toon_format import encode, decode, estimate_savings

# Encode Python → TOON
toon_str = encode(data, options={
    "delimiter": ",",      # ",", "\t", or "|"
    "indent": 2,           # Spaces per level
    "lengthMarker": "",    # "" or "#"
})

# Decode TOON → Python
data = decode(toon_str, options={
    "indent": 2,
    "strict": True,        # Enforce validation
})

# Token analysis
result = estimate_savings(data)
print(f"Saves {result['savings_percent']:.1f}% tokens")
```

### 3.4 CLI Tool

```bash
# JSON → TOON
toon input.json -o output.toon

# TOON → JSON
toon data.toon -o output.json

# With options
toon data.json --delimiter "\t" --length-marker
```

---

## 4. Integration Strategy for ccflow

### 4.1 Decision: Use Official Library

**Rationale**:
- Production-ready with 792 tests
- Actively maintained, spec-compliant
- Saves development time
- Ensures compatibility with TOON ecosystem

### 4.2 Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                    User Application                          │
│   options = CLIAgentOptions(                                 │
│       context={"positions": [...], "signals": [...]},        │
│       toon=ToonConfig(enabled=True)                          │
│   )                                                          │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              ccflow Middleware                               │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  TOON Integration Layer                               │  │
│   │  • Auto-encode context objects                        │  │
│   │  • Inject into --append-system-prompt                 │  │
│   │  • Track compression metrics                          │  │
│   └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  claude -p "prompt" --append-system-prompt "[TOON]\n..."     │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 TOON Configuration Class

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ToonConfig:
    """Configuration for TOON serialization in CLI middleware."""

    # Enable/disable TOON encoding
    enabled: bool = True

    # Delimiter: comma (default), tab (more efficient), pipe
    delimiter: Literal[",", "\t", "|"] = ","

    # Indentation spaces per level
    indent: int = 2

    # Include length markers in output
    length_marker: bool = False

    # Fields to auto-encode
    encode_context: bool = True
    encode_tool_results: bool = True
    encode_session_metadata: bool = False

    # Benchmarking
    track_savings: bool = True

    # Last compression stats (populated after encoding)
    _last_json_tokens: int = field(default=0, repr=False)
    _last_toon_tokens: int = field(default=0, repr=False)

    @property
    def last_compression_ratio(self) -> float:
        """Returns savings as fraction (0.0-1.0)."""
        if self._last_json_tokens == 0:
            return 0.0
        return 1.0 - (self._last_toon_tokens / self._last_json_tokens)
```

### 4.4 TOON Wrapper Module

```python
# claude_cli_middleware/toon_integration.py

from typing import Any
import json

# Use official library
from toon_format import encode as toon_encode, decode as toon_decode
from toon_format import estimate_savings, count_tokens

class ToonSerializer:
    """Wrapper around toon-format library for middleware integration."""

    def __init__(self, config: ToonConfig):
        self.config = config

    def encode(self, data: Any) -> str:
        """Encode Python object to TOON format."""
        if not self.config.enabled:
            return json.dumps(data, indent=2)

        options = {
            "delimiter": self.config.delimiter,
            "indent": self.config.indent,
            "lengthMarker": "#" if self.config.length_marker else "",
        }

        toon_str = toon_encode(data, options)

        if self.config.track_savings:
            self._track_savings(data, toon_str)

        return toon_str

    def decode(self, toon_str: str) -> Any:
        """Decode TOON format to Python object."""
        return toon_decode(toon_str, {"strict": True})

    def _track_savings(self, data: Any, toon_str: str) -> None:
        """Track token savings for metrics."""
        json_str = json.dumps(data)
        self.config._last_json_tokens = count_tokens(json_str)
        self.config._last_toon_tokens = count_tokens(toon_str)

    def format_for_prompt(self, data: Any, label: str = "Context") -> str:
        """Format data for injection into system prompt."""
        toon_str = self.encode(data)
        return f"\n[{label} - TOON Format]\n```toon\n{toon_str}\n```"
```

### 4.5 CLI Executor Integration

```python
# In executor.py

async def execute(
    self,
    prompt: str,
    options: CLIAgentOptions,
) -> AsyncIterator[Message]:
    """Execute CLI with TOON-encoded context."""

    flags = self._build_flags(options)

    # Inject TOON-encoded context into system prompt
    if options.context and options.toon.enabled:
        toon_serializer = ToonSerializer(options.toon)
        toon_context = toon_serializer.format_for_prompt(
            options.context,
            label="Structured Context"
        )

        # Append to existing system prompt suffix
        if "--append-system-prompt" in flags:
            idx = flags.index("--append-system-prompt")
            flags[idx + 1] += toon_context
        else:
            flags.extend(["--append-system-prompt", toon_context])

        # Log savings
        if options.toon.track_savings:
            ratio = options.toon.last_compression_ratio
            logger.info(f"TOON compression: {ratio:.1%} token savings")

    # Execute CLI subprocess
    async for msg in self._run_subprocess(prompt, flags):
        yield msg
```

---

## 5. When to Use TOON

### 5.1 Ideal Use Cases (Use TOON)

| Use Case | Token Savings | Example |
|----------|---------------|---------|
| Portfolio positions | 50-60% | `[{symbol, qty, price, pnl}]` |
| Order history | 50-60% | `[{orderId, symbol, side, qty, status}]` |
| Market data | 40-50% | `[{symbol, bid, ask, volume}]` |
| Signal arrays | 50-60% | `[{symbol, score, reason}]` |
| Config objects | 20-30% | `{setting: value}` nested |

### 5.2 Avoid TOON

| Use Case | Why |
|----------|-----|
| Deeply nested (4+ levels) | JSON may use fewer tokens |
| Non-uniform arrays | Mixed structures don't compress well |
| Small payloads (<100 tokens) | Overhead outweighs savings |
| Latency-critical paths | Encoding adds ~1-5ms |

### 5.3 Decision Matrix

```python
def should_use_toon(data: Any) -> bool:
    """Heuristic for TOON applicability."""
    if isinstance(data, list) and len(data) > 3:
        if all(isinstance(item, dict) for item in data):
            # Check uniformity
            keys = [set(item.keys()) for item in data]
            if len(set(frozenset(k) for k in keys)) == 1:
                return True  # Uniform array - TOON excels

    if isinstance(data, dict):
        # Check nesting depth
        def max_depth(obj, depth=0):
            if isinstance(obj, dict):
                return max((max_depth(v, depth+1) for v in obj.values()), default=depth)
            elif isinstance(obj, list):
                return max((max_depth(v, depth+1) for v in obj), default=depth)
            return depth

        if max_depth(data) <= 3:
            return True  # Shallow enough for TOON

    return False  # Default to JSON
```

---

## 6. Implementation Plan

### Phase 1: Add Dependency (Immediate)

```toml
# pyproject.toml
[project]
dependencies = [
    "toon-format @ git+https://github.com/toon-format/toon-python.git",
    # or when available:
    # "toon-format>=0.9.0",
]
```

### Phase 2: Create Integration Layer

1. `claude_cli_middleware/toon_integration.py` - Wrapper module
2. `claude_cli_middleware/types.py` - Add ToonConfig dataclass
3. Update `CLIAgentOptions` to include `toon: ToonConfig`

### Phase 3: Integrate with Executor

1. Modify `executor.py` to auto-encode context
2. Add TOON metrics to observability layer
3. Add CLI flag for enabling/disabling: `--toon` / `--no-toon`

### Phase 4: Testing

```python
# tests/test_toon_integration.py

def test_encode_uniform_array():
    data = [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]
    serializer = ToonSerializer(ToonConfig())
    result = serializer.encode(data)
    assert "[2]{id,name}:" in result

def test_compression_tracking():
    config = ToonConfig(track_savings=True)
    serializer = ToonSerializer(config)
    serializer.encode([{"x": 1}] * 100)
    assert config.last_compression_ratio > 0.3

def test_fallback_to_json():
    config = ToonConfig(enabled=False)
    serializer = ToonSerializer(config)
    result = serializer.encode({"a": 1})
    assert result == '{\n  "a": 1\n}'
```

---

## 7. Sources

- [GitHub: toon-format/toon](https://github.com/toon-format/toon) - TypeScript reference implementation
- [GitHub: toon-format/toon-python](https://github.com/toon-format/toon-python) - Official Python library
- [TOON Specification v3.0](https://github.com/toon-format/spec/blob/main/SPEC.md) - Formal spec
- [toonformat.dev](https://toonformat.dev/) - Official documentation
- [npm: @toon-format/toon](https://www.npmjs.com/package/@toon-format/toon) - npm package
- [DEV.to: TOON Overview](https://dev.to/abhilaksharora/toon-token-oriented-object-notation-the-smarter-lighter-json-for-llms-2f05)
- [Medium: TOON for LLMs](https://medium.com/@pablojusue/token-oriented-object-notation-toon-a-leaner-format-for-llm-data-5607c1fb6123)
- [Tensorlake: TOON vs JSON](https://www.tensorlake.ai/blog/toon-vs-json)
- [FreeCodeCamp: What is TOON](https://www.freecodecamp.org/news/what-is-toon-how-token-oriented-object-notation-could-change-how-ai-sees-data/)

---

## 8. Conclusion

TOON is a mature, well-specified format with an official Python implementation ready for production use. For the ccflow middleware:

1. **Install `toon-format`** from GitHub (or PyPI when available)
2. **Create thin wrapper** for middleware-specific concerns (metrics, prompt formatting)
3. **Auto-detect** when TOON is beneficial vs. defaulting to JSON
4. **Track savings** for observability and cost optimization

Expected impact: **40-60% reduction in context tokens** for structured data payloads, directly reducing subscription usage pressure.
