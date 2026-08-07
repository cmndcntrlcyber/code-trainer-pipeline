"""
Canonical tool schemas for nexus-harness, mirrored from:
  nexus-harness/crates/nexus-tools/src/lib.rs
  nexus-harness/crates/nexus-tools/src/todo.rs
  nexus-harness/crates/nexus-tools/src/search.rs
  nexus-harness/crates/nexus-tools/src/scope_check.rs

All eval files import from here so schemas stay in sync with the harness.
"""

NEXUS_CORE_TOOLS = [
    {"type": "function", "function": {"name": "Read", "description": "Read the contents of a UTF-8 text file at the given path (relative to the working directory).", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path to read, relative to the working directory."}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "Write", "description": "Write `content` to the file at `path` (relative to the working directory), creating parent directories and overwriting any existing file.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Destination file path."}, "content": {"type": "string", "description": "Full file contents to write."}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "Edit", "description": "Replace an exact occurrence of `old_string` with `new_string` in the file at `path`. Errors if `old_string` is absent or matches more than once, unless `replace_all` is true.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File to edit."}, "old_string": {"type": "string", "description": "Exact text to replace."}, "new_string": {"type": "string", "description": "Replacement text."}, "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring a unique match."}}, "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "LS", "description": "List the entries of a directory (defaults to the working directory). Directories are suffixed with `/`.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory to list, relative to the working directory (default `.`)."}}}}},
    {"type": "function", "function": {"name": "Bash", "description": "Run a shell command in the working directory and return its combined output. Optional `timeout_ms` bounds the run (default 120000).", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command to execute."}, "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds (default 120000)."}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "Grep", "description": "Search file contents for a regular-expression `pattern`, honoring .gitignore. Optional `path` sets the base directory. Returns `file:line:text` matches relative to the working directory.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Regular expression to search for."}, "path": {"type": "string", "description": "Base directory (default `.`)."}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "Glob", "description": "Find files whose path matches a glob `pattern` (use `**` to recurse), honoring .gitignore. Optional `path` sets the base directory. Returns matching paths relative to the working directory.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Glob, e.g. `**/*.rs`."}, "path": {"type": "string", "description": "Base directory (default `.`)."}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "WebFetch", "description": "Fetch a URL over HTTP(S) and return the response body as text (truncated). Read-only.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "URL to fetch."}}, "required": ["url"]}}},
]

NEXUS_SESSION_TOOLS = [
    {"type": "function", "function": {"name": "TodoWrite", "description": "Read or replace the session todo list. Pass `todos` (an array) to replace it; omit it to read the current list. Returns the current list.", "parameters": {"type": "object", "properties": {"todos": {"type": "array", "description": "Full replacement todo list. Each item is an object, e.g. {\"content\":\"...\",\"status\":\"pending\"}.", "items": {"type": "object"}}}}}},
    {"type": "function", "function": {"name": "Skill", "description": "Invoke a named skill: load its instructions and complete the task following them. `name` is the skill (from the skills index); optional `input` is the task/details.", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "Skill name from the skills index."}, "input": {"type": "string", "description": "Task or details for the skill (optional)."}}, "required": ["name"]}}},
    {"type": "function", "function": {"name": "Task", "description": "Delegate a task to a named subagent profile. `subagent_type` selects the profile; `prompt` is the task. Returns the subagent's final result.", "parameters": {"type": "object", "properties": {"subagent_type": {"type": "string", "description": "Subagent profile name."}, "prompt": {"type": "string", "description": "The task for the subagent."}}, "required": ["subagent_type", "prompt"]}}},
]

NEXUS_SCOPE_TOOLS = [
    {"type": "function", "function": {"name": "ScopeCheck", "description": "Check whether a target is within the authorized engagement scope. Returns ALLOW if the target matches a scope entry (and is not excluded) or DENY otherwise.", "parameters": {"type": "object", "properties": {"target": {"type": "string", "description": "The target to check: an IP address, CIDR, or domain name."}}, "required": ["target"]}}},
]

NEXUS_TOOLS_V7 = NEXUS_CORE_TOOLS[:]
NEXUS_TOOLS_V10 = NEXUS_CORE_TOOLS + NEXUS_SESSION_TOOLS + NEXUS_SCOPE_TOOLS
