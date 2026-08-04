"""Tests for tool_format_converter.py — the critical format conversion layer."""
import json

import pytest

from src.phase2_preprocessing.converters.tool_format_converter import (
    convert_fable5_messages_to_hermes,
    convert_hermes_conversations_to_messages,
    detect_tools_in_messages,
    openai_tool_calls_to_hermes_xml,
    validate_messages,
)


class TestOpenAIToolCallsToHermesXML:
    def test_single_tool_call(self):
        tool_calls = [
            {
                "id": "toolu_01ABC",
                "type": "function",
                "function": {
                    "name": "Read",
                    "arguments": '{"file_path": "main.py"}',
                },
            }
        ]
        result = openai_tool_calls_to_hermes_xml(tool_calls)
        assert "<tool_call>" in result
        assert "</tool_call>" in result
        parsed = json.loads(result.split("<tool_call>\n")[1].split("\n</tool_call>")[0])
        assert parsed["name"] == "Read"
        assert parsed["arguments"] == {"file_path": "main.py"}

    def test_multiple_tool_calls(self):
        tool_calls = [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "Read", "arguments": '{"file_path": "a.py"}'},
            },
            {
                "id": "t2",
                "type": "function",
                "function": {"name": "Write", "arguments": '{"file_path": "b.py", "content": "x"}'},
            },
        ]
        result = openai_tool_calls_to_hermes_xml(tool_calls)
        assert result.count("<tool_call>") == 2
        assert result.count("</tool_call>") == 2

    def test_string_arguments_parsed(self):
        tool_calls = [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "Bash", "arguments": '{"command": "ls -la"}'},
            }
        ]
        result = openai_tool_calls_to_hermes_xml(tool_calls)
        parsed = json.loads(result.split("<tool_call>\n")[1].split("\n</tool_call>")[0])
        assert isinstance(parsed["arguments"], dict)
        assert parsed["arguments"]["command"] == "ls -la"

    def test_dict_arguments_passed_through(self):
        tool_calls = [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "Grep", "arguments": {"pattern": "TODO"}},
            }
        ]
        result = openai_tool_calls_to_hermes_xml(tool_calls)
        parsed = json.loads(result.split("<tool_call>\n")[1].split("\n</tool_call>")[0])
        assert parsed["arguments"] == {"pattern": "TODO"}

    def test_empty_arguments(self):
        tool_calls = [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "GetStatus", "arguments": "{}"},
            }
        ]
        result = openai_tool_calls_to_hermes_xml(tool_calls)
        parsed = json.loads(result.split("<tool_call>\n")[1].split("\n</tool_call>")[0])
        assert parsed["arguments"] == {}

    def test_malformed_arguments_falls_back(self):
        tool_calls = [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "Bad", "arguments": "not valid json{"},
            }
        ]
        result = openai_tool_calls_to_hermes_xml(tool_calls)
        assert "<tool_call>" in result
        parsed = json.loads(result.split("<tool_call>\n")[1].split("\n</tool_call>")[0])
        assert parsed["name"] == "Bad"
        assert parsed["arguments"] == "not valid json{"


class TestHermesConversionsToMessages:
    def test_role_mapping(self):
        conversations = [
            {"from": "system", "value": "You are helpful."},
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there!"},
        ]
        result = convert_hermes_conversations_to_messages(conversations)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"

    def test_tool_role_preserved(self):
        conversations = [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "call tool"},
            {"from": "gpt", "value": "<tool_call>\n{}\n</tool_call>"},
            {"from": "tool", "value": "<tool_response>\nresult\n</tool_response>"},
            {"from": "gpt", "value": "Done."},
        ]
        result = convert_hermes_conversations_to_messages(conversations)
        assert len(result) == 5
        assert result[3]["role"] == "tool"
        assert result[4]["role"] == "assistant"

    def test_content_preserved(self):
        conversations = [
            {"from": "system", "value": "You have <tools>[...]</tools>"},
            {"from": "human", "value": "What's the weather?"},
        ]
        result = convert_hermes_conversations_to_messages(conversations)
        assert "<tools>" in result[0]["content"]
        assert result[1]["content"] == "What's the weather?"


class TestFable5MessagesToHermes:
    def test_assistant_with_tool_calls(self):
        messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Read the file."},
            {
                "role": "assistant",
                "content": "I'll read it.",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": '{"file_path": "x.py"}'},
                    }
                ],
                "tool_call_id": "",
                "name": "",
            },
        ]
        result = convert_fable5_messages_to_hermes(messages)
        assert result[2]["role"] == "assistant"
        assert result[2]["content"].startswith("I'll read it.\n<tool_call>")
        assert '"name": "Read"' in result[2]["content"]

    def test_empty_content_with_tool_calls(self):
        messages = [
            {"role": "user", "content": "Do it."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
                    }
                ],
            },
        ]
        result = convert_fable5_messages_to_hermes(messages)
        assert result[1]["content"].startswith("<tool_call>")
        assert not result[1]["content"].startswith("\n")

    def test_tool_result_wrapped(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            {
                "role": "tool",
                "content": "file contents here",
                "tool_call_id": "t1",
            },
        ]
        result = convert_fable5_messages_to_hermes(messages)
        assert result[2]["role"] == "tool"
        assert result[2]["content"] == "<tool_response>\nfile contents here\n</tool_response>"

    def test_reasoning_content_discarded(self):
        messages = [
            {"role": "user", "content": "think about this"},
            {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning_content": "Let me think step by step...",
                "tool_calls": [],
            },
        ]
        result = convert_fable5_messages_to_hermes(messages)
        assert "step by step" not in result[1]["content"]
        assert result[1]["content"] == "The answer is 42."

    def test_none_content_handled(self):
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": '{"path": "/"}'},
                    }
                ],
            },
        ]
        result = convert_fable5_messages_to_hermes(messages)
        assert result[1]["content"].startswith("<tool_call>")

    def test_multi_turn_conversation(self):
        messages = [
            {"role": "system", "content": "Agent mode."},
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "Let me look.",
                "tool_calls": [
                    {"id": "t1", "type": "function", "function": {"name": "Read", "arguments": '{"file_path": "bug.py"}'}},
                ],
            },
            {"role": "tool", "content": "def broken(): pass", "tool_call_id": "t1"},
            {"role": "assistant", "content": "Found it. The function is empty."},
        ]
        result = convert_fable5_messages_to_hermes(messages)
        assert len(result) == 5
        assert "<tool_call>" in result[2]["content"]
        assert "<tool_response>" in result[3]["content"]
        assert result[4]["content"] == "Found it. The function is empty."


class TestDetectTools:
    def test_detects_tool_call(self):
        messages = [
            {"role": "assistant", "content": "<tool_call>\n{}\n</tool_call>"},
        ]
        assert detect_tools_in_messages(messages) is True

    def test_detects_tools_definition(self):
        messages = [
            {"role": "system", "content": "You have <tools>[...]</tools>"},
        ]
        assert detect_tools_in_messages(messages) is True

    def test_no_tools(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert detect_tools_in_messages(messages) is False


class TestValidateMessages:
    def test_valid_minimal(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert validate_messages(messages) is True

    def test_valid_with_system(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert validate_messages(messages) is True

    def test_too_few_messages(self):
        assert validate_messages([{"role": "user", "content": "hi"}]) is False

    def test_no_assistant(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "hello?"},
        ]
        assert validate_messages(messages) is False

    def test_missing_role_key(self):
        messages = [
            {"content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert validate_messages(messages) is False

    def test_none_content_fails(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None},
        ]
        assert validate_messages(messages) is False
