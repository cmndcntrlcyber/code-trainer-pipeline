"""
phase4_qwen_finetuning/hf_skills/agent_eval_entry.py

Agent behavior validation for V7 corrective action. Runs 5 multi-turn
agent scenarios, checking whether the model reasons, calls tools,
interprets results, and makes progress on coding tasks.

Target: >= 3/5 scenarios making progress.

Expected env vars:
    HF_TOKEN                          — read base/adapter, push result
    PHASE4_AGENT_EVAL_PARAMS_JSON     — { model_id, adapter_repo, upload_repo,
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

AGENT_TOOLS = [
    {"type": "function", "function": {"name": "Read", "description": "Read a file from the filesystem", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}},
    {"type": "function", "function": {"name": "Write", "description": "Write content to a file", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}}},
    {"type": "function", "function": {"name": "Bash", "description": "Execute a bash command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "Grep", "description": "Search for a pattern in files", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
]

SYSTEM_PROMPT_TEMPLATE = (
    "You are a coding assistant with access to tools. "
    "When you need to perform actions, use the tools available to you "
    "by emitting <tool_call> XML tags. Think step by step about what "
    "needs to be done before acting.\n\n"
    "<tools>\n{tools_json}\n</tools>"
)

SCENARIOS = [
    {
        "name": "read_and_analyze",
        "description": "Read a file and explain what it does",
        "turns": [
            {"role": "user", "content": "Read the file src/main.py and tell me what it does."},
            {"role": "tool", "content": "<tool_response>\nimport argparse\nimport sys\n\ndef parse_args():\n    parser = argparse.ArgumentParser(description='Data processor')\n    parser.add_argument('--input', required=True, help='Input CSV file')\n    parser.add_argument('--output', default='result.json', help='Output file')\n    return parser.parse_args()\n\ndef main():\n    args = parse_args()\n    print(f'Processing {args.input} -> {args.output}')\n\nif __name__ == '__main__':\n    main()\n</tool_response>"},
        ],
        "success_criteria": ["calls Read tool", "explains the code"],
    },
    {
        "name": "find_and_fix_bug",
        "description": "Find a bug in code and propose a fix",
        "turns": [
            {"role": "user", "content": "There's a bug in utils.py where division by zero can happen. Find and fix it."},
            {"role": "tool", "content": "<tool_response>\ndef calculate_average(numbers):\n    total = sum(numbers)\n    return total / len(numbers)\n\ndef normalize(values):\n    max_val = max(values)\n    min_val = min(values)\n    return [(v - min_val) / (max_val - min_val) for v in values]\n</tool_response>"},
        ],
        "success_criteria": ["identifies division by zero", "proposes a fix"],
    },
    {
        "name": "multi_file_investigation",
        "description": "Investigate across multiple files",
        "turns": [
            {"role": "user", "content": "Find all files that import the 'requests' library and list them."},
            {"role": "tool", "content": "<tool_response>\nsrc/api_client.py:1:import requests\nsrc/scraper.py:3:import requests\ntests/test_api.py:2:from unittest.mock import patch\ntests/test_api.py:3:import requests\n</tool_response>"},
        ],
        "success_criteria": ["calls Grep or Bash", "lists the files found"],
    },
    {
        "name": "write_new_code",
        "description": "Write a new function based on requirements",
        "turns": [
            {"role": "user", "content": "Write a Python function in utils.py that validates email addresses using a regex pattern. It should return True for valid emails and False otherwise."},
        ],
        "success_criteria": ["calls Write tool", "includes regex pattern", "returns boolean"],
    },
    {
        "name": "run_and_debug",
        "description": "Run a command and interpret the output",
        "turns": [
            {"role": "user", "content": "Run the test suite and tell me what's failing."},
            {"role": "tool", "content": "<tool_response>\n============================= test session starts ==============================\ncollected 12 items\n\ntests/test_parser.py::test_parse_valid PASSED\ntests/test_parser.py::test_parse_empty PASSED\ntests/test_api.py::test_get_user FAILED - AssertionError: expected status 200, got 404\ntests/test_api.py::test_create_user PASSED\ntests/test_api.py::test_delete_user FAILED - ConnectionError: refused\ntests/test_utils.py::test_average PASSED\n\n===================== 2 failed, 4 passed in 1.23s ============================\n</tool_response>"},
        ],
        "success_criteria": ["calls Bash tool", "identifies failing tests"],
    },
]


def check_progress(conversation: list[dict], scenario: dict) -> dict:
    """Check whether the model made progress on the scenario."""
    assistant_turns = [m for m in conversation if m["role"] == "assistant"]
    if not assistant_turns:
        return {"making_progress": False, "reason": "no assistant response"}

    all_text = " ".join(m["content"] for m in assistant_turns)

    used_tools = bool(TOOL_CALL_PATTERN.search(all_text))
    gave_explanation = len(all_text) > 50
    not_refusing = not any(
        phrase in all_text.lower()
        for phrase in ["i cannot", "i'm unable", "i don't have access", "as an ai"]
    )
    not_looping = len(set(m["content"][:100] for m in assistant_turns)) == len(assistant_turns)

    making_progress = (used_tools or gave_explanation) and not_refusing and not_looping

    reasons = []
    if used_tools:
        reasons.append("used tools")
    if gave_explanation:
        reasons.append("gave explanation")
    if not not_refusing:
        reasons.append("refused the task")
    if not not_looping:
        reasons.append("looping/repeating")

    return {
        "making_progress": making_progress,
        "used_tools": used_tools,
        "gave_explanation": gave_explanation,
        "not_refusing": not_refusing,
        "not_looping": not_looping,
        "reason": "; ".join(reasons),
    }


def main():
    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    params = json.loads(os.environ.get("PHASE4_AGENT_EVAL_PARAMS_JSON", "{}"))
    model_id = params.get("model_id", "Qwen/Qwen2.5-Coder-14B-Instruct")
    adapter_repo = params.get("adapter_repo")
    upload_repo = params.get("upload_repo", adapter_repo)
    result_filename = params.get("result_filename", "phase4-agent-eval.json")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN required")
    if not upload_repo:
        raise RuntimeError("upload_repo required")

    logger.info("=" * 60)
    logger.info("PHASE 4 — V7 Agent Behavior Evaluation")
    logger.info(f"  base:     {model_id}")
    logger.info(f"  adapter:  {adapter_repo or '(baseline)'}")
    logger.info(f"  upload:   {upload_repo}")
    logger.info(f"  scenarios: {len(SCENARIOS)}")
    logger.info("=" * 60)

    tools_json = json.dumps(AGENT_TOOLS, indent=2)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(tools_json=tools_json)

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

        conversation = [{"role": "system", "content": system_prompt}]

        for turn in scenario["turns"]:
            conversation.append(turn)

            if turn["role"] in ("user", "tool"):
                text = tokenizer.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True,
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,
                        temperature=1.0,
                    )
                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                conversation.append({"role": "assistant", "content": response})
                logger.info("  Turn %d response length: %d chars",
                            len(conversation) - 1, len(response))

        progress = check_progress(conversation, scenario)
        results.append({
            "scenario": scenario["name"],
            "description": scenario["description"],
            **progress,
            "n_turns": len(conversation),
        })
        logger.info("  %s: %s — %s", scenario["name"],
                     "PROGRESS" if progress["making_progress"] else "NO PROGRESS",
                     progress["reason"])

    progressing = sum(1 for r in results if r["making_progress"])
    total = len(results)

    payload = {
        "model": model_id,
        "adapter": adapter_repo,
        "scenarios": total,
        "progressing": progressing,
        "progress_rate": progressing / total,
        "target_threshold": 3,
        "meets_target": progressing >= 3,
        "results": results,
    }

    logger.info("=" * 60)
    logger.info("Agent eval: %d/%d making progress — %s",
                progressing, total,
                "PASS" if payload["meets_target"] else "FAIL")
    logger.info("=" * 60)

    out_path = Path("/tmp/phase4-agent-eval") / result_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=result_filename,
        repo_id=upload_repo,
        repo_type="model",
        commit_message=f"V7 agent eval: {progressing}/{total} making progress",
    )
    logger.info("Result pushed: https://huggingface.co/%s/blob/main/%s",
                upload_repo, result_filename)


if __name__ == "__main__":
    main()
