"""
phase4_qwen_finetuning/hf_skills/tool_call_eval_entry.py

Tool-calling validation for V7 corrective action. Presents the model with
10 scenarios containing <tools> definitions, verifies it emits valid
<tool_call> XML output with correct structure and parseable JSON arguments.

Target: >= 80% pass rate (8/10 scenarios).

Expected env vars:
    HF_TOKEN                         — read base/adapter, push result
    PHASE4_TOOL_EVAL_PARAMS_JSON     — { model_id, adapter_repo, upload_repo,
                                         result_filename }
"""
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HOME", "/workspace/.hf-cache")

TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

SCENARIOS = [
    {
        "name": "simple_function_call",
        "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
        "user": "What's the weather like in Seattle?",
        "expected_tool": "get_weather",
    },
    {
        "name": "multiple_parameters",
        "tools": [{"type": "function", "function": {"name": "search_flights", "description": "Search for flights", "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}}}],
        "user": "Find flights from New York to London on December 25th.",
        "expected_tool": "search_flights",
    },
    {
        "name": "nested_object_parameters",
        "tools": [{"type": "function", "function": {"name": "create_user", "description": "Create a new user account", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "address": {"type": "object", "properties": {"street": {"type": "string"}, "city": {"type": "string"}, "zip": {"type": "string"}}}}, "required": ["name"]}}}],
        "user": "Create a user named Alice at 123 Main St, Portland, 97201.",
        "expected_tool": "create_user",
    },
    {
        "name": "choose_between_two_tools",
        "tools": [
            {"type": "function", "function": {"name": "send_email", "description": "Send an email", "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}},
            {"type": "function", "function": {"name": "send_sms", "description": "Send an SMS message", "parameters": {"type": "object", "properties": {"phone": {"type": "string"}, "message": {"type": "string"}}, "required": ["phone", "message"]}}},
        ],
        "user": "Text 555-0123 and say 'meeting at 3pm'.",
        "expected_tool": "send_sms",
    },
    {
        "name": "choose_between_three_tools",
        "tools": [
            {"type": "function", "function": {"name": "read_file", "description": "Read a file from disk", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write content to a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "delete_file", "description": "Delete a file from disk", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        ],
        "user": "Show me the contents of /etc/hostname.",
        "expected_tool": "read_file",
    },
    {
        "name": "tool_with_optional_params",
        "tools": [{"type": "function", "function": {"name": "list_processes", "description": "List running processes", "parameters": {"type": "object", "properties": {"filter": {"type": "string", "description": "Optional name filter"}, "limit": {"type": "integer", "description": "Max results"}}, "required": []}}}],
        "user": "Show me all running python processes.",
        "expected_tool": "list_processes",
    },
    {
        "name": "tool_with_enum_type",
        "tools": [{"type": "function", "function": {"name": "set_priority", "description": "Set task priority", "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}, "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]}}, "required": ["task_id", "priority"]}}}],
        "user": "Set task 42 to critical priority.",
        "expected_tool": "set_priority",
    },
    {
        "name": "tool_with_array_parameter",
        "tools": [{"type": "function", "function": {"name": "tag_items", "description": "Add tags to items", "parameters": {"type": "object", "properties": {"item_ids": {"type": "array", "items": {"type": "integer"}}, "tag": {"type": "string"}}, "required": ["item_ids", "tag"]}}}],
        "user": "Tag items 1, 2, and 3 as 'urgent'.",
        "expected_tool": "tag_items",
    },
    {
        "name": "no_tool_needed",
        "tools": [{"type": "function", "function": {"name": "calculator", "description": "Perform math calculations", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}],
        "user": "What is the capital of France?",
        "expected_tool": None,
    },
    {
        "name": "sequential_tool_calls",
        "tools": [
            {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "grep", "description": "Search for pattern in text", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "text": {"type": "string"}}, "required": ["pattern", "text"]}}},
        ],
        "user": "Read the file config.yaml and find all lines containing 'timeout'.",
        "expected_tool": "read_file",
    },
]


def build_system_prompt(tools: list[dict]) -> str:
    tools_json = json.dumps(tools, indent=2)
    return (
        "You are a helpful assistant with access to the following tools. "
        "Use them when needed by emitting <tool_call> XML tags.\n\n"
        f"<tools>\n{tools_json}\n</tools>"
    )


def evaluate_response(response: str, scenario: dict) -> dict:
    """Evaluate a single model response against a scenario."""
    expected_tool = scenario["expected_tool"]
    matches = TOOL_CALL_PATTERN.findall(response)

    if expected_tool is None:
        passed = len(matches) == 0
        return {
            "scenario": scenario["name"],
            "passed": passed,
            "reason": "correctly avoided tool call" if passed else "hallucinated a tool call",
            "response_preview": response[:200],
        }

    if not matches:
        return {
            "scenario": scenario["name"],
            "passed": False,
            "reason": "no <tool_call> found in response",
            "response_preview": response[:200],
        }

    try:
        call = json.loads(matches[0])
    except json.JSONDecodeError:
        return {
            "scenario": scenario["name"],
            "passed": False,
            "reason": "tool_call JSON is malformed",
            "response_preview": response[:200],
        }

    correct_name = call.get("name") == expected_tool
    has_arguments = "arguments" in call

    passed = correct_name and has_arguments
    reason = []
    if not correct_name:
        reason.append(f"wrong tool: got {call.get('name')}, expected {expected_tool}")
    if not has_arguments:
        reason.append("missing arguments key")

    return {
        "scenario": scenario["name"],
        "passed": passed,
        "reason": "; ".join(reason) if reason else "correct tool call",
        "tool_called": call.get("name"),
        "response_preview": response[:200],
    }


def main():
    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    params = json.loads(os.environ.get("PHASE4_TOOL_EVAL_PARAMS_JSON", "{}"))
    model_id = params.get("model_id", "Qwen/Qwen2.5-Coder-14B-Instruct")
    adapter_repo = params.get("adapter_repo")
    upload_repo = params.get("upload_repo", adapter_repo)
    result_filename = params.get("result_filename", "phase4-tool-call-eval.json")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required")
    if not upload_repo:
        raise RuntimeError("upload_repo required")

    logger.info("=" * 60)
    logger.info("PHASE 4 — V7 Tool-Calling Evaluation")
    logger.info(f"  base:     {model_id}")
    logger.info(f"  adapter:  {adapter_repo or '(baseline)'}")
    logger.info(f"  upload:   {upload_repo}")
    logger.info(f"  scenarios: {len(SCENARIOS)}")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if adapter_repo:
        logger.info("Loading adapter: %s", adapter_repo)
        model = PeftModel.from_pretrained(model, adapter_repo)
        model = model.merge_and_unload()

    model.eval()

    results = []
    for i, scenario in enumerate(SCENARIOS):
        logger.info("Scenario %d/%d: %s", i + 1, len(SCENARIOS), scenario["name"])
        messages = [
            {"role": "system", "content": build_system_prompt(scenario["tools"])},
            {"role": "user", "content": scenario["user"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        result = evaluate_response(response, scenario)
        results.append(result)
        logger.info("  %s: %s — %s", scenario["name"],
                     "PASS" if result["passed"] else "FAIL", result["reason"])

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_rate = passed / total

    payload = {
        "model": model_id,
        "adapter": adapter_repo,
        "scenarios": total,
        "passed": passed,
        "pass_rate": pass_rate,
        "target_pass_rate": 0.80,
        "meets_target": pass_rate >= 0.80,
        "results": results,
    }

    logger.info("=" * 60)
    logger.info("Tool-calling eval: %d/%d passed (%.0f%%) — %s",
                passed, total, pass_rate * 100,
                "PASS" if payload["meets_target"] else "FAIL")
    logger.info("=" * 60)

    out_path = Path("/tmp/phase4-tool-eval") / result_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=result_filename,
        repo_id=upload_repo,
        repo_type="model",
        commit_message=f"V7 tool-calling eval: {passed}/{total} ({pass_rate:.0%})",
    )
    logger.info("Result pushed: https://huggingface.co/%s/blob/main/%s",
                upload_repo, result_filename)


if __name__ == "__main__":
    main()
