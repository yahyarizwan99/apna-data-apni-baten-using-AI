"""The AI side of Apna Data Apni Baten.

Works with either OpenAI or Anthropic. The model never guesses a number — it writes
pandas code, we run it, and it answers from the real output. It can also read this
app's own source, so you can ask how the app itself works.
"""

from __future__ import annotations

import ast
import io
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

MAX_STEPS = 8
MAX_OUTPUT_CHARS = 6000

# price is USD per 1M tokens (input/output), rough — check the provider for current rates
PROVIDERS = {
    "OpenAI": {
        "models": [
            ("gpt-4.1-nano",  "cheapest  ~$0.10 / $0.40"),
            ("gpt-4o-mini",   "cheap     ~$0.15 / $0.60"),
            ("gpt-4.1-mini",  "balanced  ~$0.40 / $1.60"),
            ("gpt-3.5-turbo", "old, weak ~$0.50 / $1.50"),
            ("gpt-4.1",       "strong    ~$2.00 / $8.00"),
            ("gpt-4o",        "strong    ~$2.50 / $10.00"),
        ],
        "keys_url": "https://platform.openai.com/api-keys",
        "prices_url": "https://openai.com/api/pricing",
        "prefix": "sk-...",
        "secret": "OPENAI_API_KEY",
    },
    "Anthropic": {
        "models": [
            ("claude-haiku-4-5-20251001", "cheapest and fast"),
            ("claude-sonnet-5-5",         "balanced"),
            ("claude-opus-5-5",           "strongest, priciest"),
        ],
        "keys_url": "https://console.anthropic.com/settings/keys",
        "prices_url": "https://www.anthropic.com/pricing",
        "prefix": "sk-ant-...",
        "secret": "ANTHROPIC_API_KEY",
    },
}

# files the AI is allowed to read when asked about itself
SOURCE_FILES = ["app.py", "ai_agent.py", "data_tools.py", "requirements.txt", "README.md"]

SYSTEM = """You are {ai_name}, a data analyst built into an app called Apna Data Apni Baten. \
You are talking to the person who loaded this dataset. Be warm, direct and short.

HOW YOU WORK
- You never guess a number. To answer anything factual about the data you call python_analysis \
and read the result.
- The DataFrame is already in scope as `df`. pandas is `pd`, numpy is `np`, plotly express is \
`px`, plotly graph objects is `go`.
- print() whatever you want to see. The last expression on its own line is also shown, \
like a notebook. Only that text comes back to you — never the whole dataset.
- For a chart, build a plotly figure and assign it to a variable whose name starts with `fig`. \
The user sees it automatically. Never call fig.show().
- If your code errors, read the error and try again with a fix.
- When you have the numbers, stop calling tools and write the answer.

QUESTIONS ABOUT THE APP ITSELF
- If asked how this app works, how the profiling is done, what you are running on, or anything \
else about the code, call read_source to read the actual file before answering. Readable \
files: {files}.
- Explain what the code really does. Never invent behaviour that isn't in the source.

HOW YOU ANSWER
- Lead with the answer. Then the few numbers that back it up.
- Markdown. Under about 150 words unless asked for more.
- Format money and large numbers readably.
- Reply in whatever language the question was asked in — English, Urdu, or Roman Urdu.
- If the data cannot answer the question, say so plainly and say what would be needed.

THE DATA
{profile}
"""

TOOL_SPECS = [
    {
        "name": "python_analysis",
        "description": "Run Python against the loaded DataFrame `df` and return whatever it prints.",
        "schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code. Use print() to return values."}
            },
            "required": ["code"],
        },
    },
    {
        "name": "read_source",
        "description": "Read one of this app's own source files, to answer questions about how the app works.",
        "schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "One of: " + ", ".join(SOURCE_FILES)}
            },
            "required": ["filename"],
        },
    },
]


class FriendlyError(Exception):
    """An error worth showing the user as-is."""


def _readable(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if any(w in low for w in ("api key", "authentication", "401", "invalid_api_key")):
        return "That API key was rejected. Check it's copied in full, has no spaces, and is still active."
    if "rate limit" in low or "429" in low:
        return "Rate limited, or your quota is used up. Wait a minute, or add credit to your account."
    if "model" in low and any(w in low for w in ("not found", "does not exist", "404", "not_found")):
        return "Your key doesn't have access to that model. Pick a different one from the list."
    if "credit" in low or "billing" in low:
        return "Your account needs billing credit before it can answer questions."
    if "connection" in low or "timeout" in low:
        return "Couldn't reach the API. Check your internet connection."
    return msg


# ---------------------------------------------------------------- tools
def _as_text(value) -> str:
    """Render a result for the model: tables as markdown, capped."""
    if isinstance(value, pd.DataFrame):
        extra = f"\n... ({len(value):,} rows total)" if len(value) > 50 else ""
        return value.head(50).to_markdown() + extra
    if isinstance(value, pd.Series):
        extra = f"\n... ({len(value):,} rows total)" if len(value) > 50 else ""
        return value.head(50).to_frame().to_markdown() + extra
    if isinstance(value, go.Figure):
        return "[chart created and shown to the user]"
    return repr(value)


def run_code(code: str, df: pd.DataFrame) -> tuple[str, list]:
    """Execute model-written code. Returns (text for the model, plotly figures).

    The last expression is evaluated and shown, the way a notebook would.
    print() is captured per call rather than by redirecting stdout, so two people
    using the app at once can't cross wires.
    """
    buf = io.StringIO()

    def _print(*args, **kwargs):
        kwargs.pop("file", None)
        print(*args, file=buf, **kwargs)

    scope = {"df": df.copy(), "pd": pd, "np": np, "px": px, "go": go, "print": _print}

    try:
        tree = ast.parse(code)
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, "<analysis>", "exec"), scope)
        result = (eval(compile(last, "<analysis>", "eval"), scope)
                  if last is not None else scope.get("result"))
    except Exception as exc:
        before = buf.getvalue()[:1500]
        return (f"ERROR: {type(exc).__name__}: {exc}"
                + (f"\nOutput before the error:\n{before}" if before else "")), []

    figures = [v for k, v in scope.items()
               if isinstance(v, go.Figure) and not k.startswith("_")]
    if isinstance(result, go.Figure) and not any(result is f for f in figures):
        figures.append(result)

    parts = []
    if buf.getvalue():
        parts.append(buf.getvalue().rstrip())
    if result is not None:
        parts.append(_as_text(result))
    if figures:
        parts.append(f"{len(figures)} chart(s) shown to the user.")

    out = "\n\n".join(parts) or "The code ran but produced no output. print() a value or leave an expression on the last line."
    if len(out) > MAX_OUTPUT_CHARS:
        out = out[:MAX_OUTPUT_CHARS] + "\n... (output cut short)"
    return out, figures


def read_source(filename: str) -> str:
    """Hand the model one of this app's own files."""
    name = Path(str(filename)).name
    if name not in SOURCE_FILES:
        return f"Not allowed. Readable files: {', '.join(SOURCE_FILES)}"
    path = Path(__file__).parent / name
    if not path.exists():
        return f"{name} isn't in the app folder."
    text = path.read_text(encoding="utf-8")
    if len(text) > 18000:
        text = text[:18000] + "\n...(file truncated)"
    return f"--- {name} ---\n{text}"


def _dispatch(name: str, args: dict, df) -> tuple[str, list, str]:
    """Returns (output, figures, text to show the user)."""
    if name == "python_analysis":
        code = args.get("code", "")
        out, figs = run_code(code, df)
        return out, figs, code
    if name == "read_source":
        fn = args.get("filename", "")
        return read_source(fn), [], f"# reading {fn}"
    return f"Unknown tool: {name}", [], ""


def _gave_up() -> str:
    return ("That took too many steps without landing on an answer. Try asking it more "
            "specifically, or split it into two questions.")


# ---------------------------------------------------------------- entry point
def ask(question, history, df, profile_text, api_key, model, ai_name, provider="OpenAI"):
    """Generator. Yields ('step', {...}) while working, then ('answer', {...}) once."""
    system = SYSTEM.format(ai_name=ai_name, profile=profile_text, files=", ".join(SOURCE_FILES))
    runner = _anthropic_loop if provider == "Anthropic" else _openai_loop
    yield from runner(question, history, df, system, api_key, model)


def _openai_loop(question, history, df, system, api_key, model):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        raise FriendlyError("The `openai` package isn't installed. Run: pip install openai")
    except Exception as exc:
        raise FriendlyError(_readable(exc)) from exc

    tools = [{"type": "function",
              "function": {"name": t["name"], "description": t["description"],
                           "parameters": t["schema"]}} for t in TOOL_SPECS]

    messages = [{"role": "system", "content": system}]
    messages += [{"role": t["role"], "content": t["content"]} for t in history[-12:]]
    messages.append({"role": "user", "content": question})

    figures, steps = [], []

    for _ in range(MAX_STEPS):
        try:
            reply = client.chat.completions.create(
                model=model, messages=messages, tools=tools, temperature=0)
        except Exception as exc:
            raise FriendlyError(_readable(exc)) from exc

        msg = reply.choices[0].message

        if not msg.tool_calls:
            yield "answer", {"content": msg.content or "I couldn't put an answer together for that one.",
                             "figures": figures, "steps": steps}
            return

        messages.append({
            "role": "assistant", "content": msg.content,
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name,
                                         "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            out, figs, shown = _dispatch(tc.function.name, args, df)
            figures.extend(figs)
            step = {"code": shown, "output": out, "charts": len(figs)}
            steps.append(step)
            yield "step", step
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})

    yield "answer", {"content": _gave_up(), "figures": figures, "steps": steps}


def _anthropic_loop(question, history, df, system, api_key, model):
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise FriendlyError("The `anthropic` package isn't installed. Run: pip install anthropic")
    except Exception as exc:
        raise FriendlyError(_readable(exc)) from exc

    tools = [{"name": t["name"], "description": t["description"],
              "input_schema": t["schema"]} for t in TOOL_SPECS]

    messages = [{"role": t["role"], "content": t["content"]} for t in history[-12:]]
    messages.append({"role": "user", "content": question})

    figures, steps = [], []

    for _ in range(MAX_STEPS):
        try:
            reply = client.messages.create(
                model=model, system=system, messages=messages,
                tools=tools, max_tokens=2048, temperature=0)
        except Exception as exc:
            raise FriendlyError(_readable(exc)) from exc

        blocks = list(reply.content)
        tool_uses = [b for b in blocks if getattr(b, "type", "") == "tool_use"]
        said = "".join(b.text for b in blocks if getattr(b, "type", "") == "text")

        if not tool_uses:
            yield "answer", {"content": said or "I couldn't put an answer together for that one.",
                             "figures": figures, "steps": steps}
            return

        messages.append({"role": "assistant", "content": blocks})

        results = []
        for block in tool_uses:
            out, figs, shown = _dispatch(block.name, dict(block.input or {}), df)
            figures.extend(figs)
            step = {"code": shown, "output": out, "charts": len(figs)}
            steps.append(step)
            yield "step", step
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})

        messages.append({"role": "user", "content": results})

    yield "answer", {"content": _gave_up(), "figures": figures, "steps": steps}
