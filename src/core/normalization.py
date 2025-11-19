import json
import re
from typing import Any, Dict, Tuple

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
FENCED_JSON_PATTERN = re.compile(r"```json(.*?)```", re.DOTALL)


def extract_reasoning_and_answer(text: str) -> Tuple[str, str]:
    think_match = THINK_PATTERN.search(text)
    reasoning = think_match.group(1).strip() if think_match else ""

    cleaned = THINK_PATTERN.sub("", text).strip()

    answer_match = ANSWER_PATTERN.search(cleaned)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = cleaned.strip()

    return reasoning, answer


def extract_tool_json(text: str) -> Dict[str, Any] | None:
    for block in FENCED_JSON_PATTERN.findall(text):
        try:
            return json.loads(block.strip())
        except Exception:
            pass

    try:
        return json.loads(text)
    except Exception:
        return None


def normalize_output(raw: str) -> Dict[str, Any]:
    reasoning, answer = extract_reasoning_and_answer(raw)
    tool_call = extract_tool_json(answer)

    return {
        "reasoning": reasoning,
        "final": answer if tool_call is None else "",
        "tool": tool_call,
    }
