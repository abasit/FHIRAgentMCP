"""
MCP-based purple agent for FHIR evaluation.

Connects to an MCP server to access FHIR tools and answers medical questions
by iteratively calling tools and reasoning over results.
"""

import json
import logging
import re
import warnings
from typing import Any, Optional

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text
from litellm import completion
from pydantic import BaseModel, Field

from mcp_client import MCPClient


warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

logger = logging.getLogger("mcp_purple_agent")

SYSTEM_PROMPT = """You are a helpful AI assistant that can complete tasks using available tools.

Use the available MCP tools to retrieve information before answering.
Provide clear, accurate answers based on the data you retrieve.
If you cannot find information, state this clearly rather than guessing.
Do not repeat the same action multiple times.

Respond in JSON format wrapped in <json>...</json> tags:

For tool calls:
<json>
{"name": "tool_name", "kwargs": {"arg1": "value1"}}
</json>

For final answer:
<json>
{"name": "response", "kwargs": {"content": "The final answer is: ..."}}
</json>

IMPORTANT: Your final answer must start with 'The final answer is:'
"""

MAX_ITERATIONS = 10
DEFAULT_MODEL = "openai/gpt-4o-mini"


class MCPContextState(BaseModel):
    """State for an MCP connection context."""
    model_config = {"arbitrary_types_allowed": True}

    url: str
    client: MCPClient
    messages: list[dict[str, str]] = Field(default_factory=list)
    tools_index: set[str] = Field(default_factory=set)
    session_id: Optional[str] = None


class Agent:
    """Purple agent that uses MCP tools to answer questions."""

    def __init__(self):
        self.ctx_id_to_state: dict[str, MCPContextState] = {}

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Process message and respond using MCP tools."""
        user_input = get_message_text(message)
        ctx_id = message.context_id

        # Extract task ID early for consistent logging
        _, task_id = self._extract_mcp_url(user_input)
        log_id = task_id or ctx_id

        logger.info(f"[Task {log_id}] Received task")
        logger.debug(f"[Task {log_id}] Input:\n{user_input[:500]}...")

        try:
            state = await self._ensure_state(ctx_id, user_input)
            logger.debug(f"[Task {log_id}] Connected to MCP at {state.url}")
        except Exception as e:
            logger.error(f"[Task {log_id}] Setup error: {e}")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error: {str(e)}"))],
                name="Error",
            )
            return

        state.messages.append({"role": "user", "content": user_input})

        assistant_content = ""
        for i in range(MAX_ITERATIONS):
            logger.debug(f"[Task {log_id}] Iteration {i + 1}/{MAX_ITERATIONS}")

            response = completion(
                messages=state.messages,
                model=DEFAULT_MODEL,
                temperature=0.0,
                top_p=0,
                seed=0,
            )
            assistant_content = response.choices[0].message.content or ""
            state.messages.append({"role": "assistant", "content": assistant_content})

            logger.debug(f"[Task {log_id}] LLM response:\n{assistant_content}")

            try:
                action = self._parse_action(assistant_content)
                logger.debug(f"[Task {log_id}] Action: {action.get('name') if action else None}")
            except Exception as e:
                logger.warning(f"[Task {log_id}] Failed to parse response: {e}")
                break

            if not action:
                logger.warning(f"[Task {log_id}] No action found in response")
                break

            name = action.get("name")
            kwargs = action.get("kwargs", {})

            # Final response
            if name == "response":
                logger.debug(f"[Task {log_id}] Got final response")
                break

            # Unknown tool
            if name not in state.tools_index:
                logger.warning(f"[Task {log_id}] Unknown tool: {name}")
                state.messages.append({
                    "role": "user",
                    "content": f"Error: Unknown tool '{name}'. Available tools: {sorted(state.tools_index)}"
                })
                continue

            # Call tool
            logger.debug(f"[Task {log_id}] Calling tool: {name}, args: {kwargs}")
            try:
                result = await state.client.call_tool(name, kwargs)
                result_text = self._format_tool_result(result)
                logger.debug(f"[Task {log_id}] Tool result:\n{result_text}")
                state.messages.append({
                    "role": "user",
                    "content": f"Tool `{name}` result:\n{result_text}"
                })
            except Exception as e:
                logger.error(f"[Task {log_id}] Tool {name} failed: {e}")
                state.messages.append({
                    "role": "user",
                    "content": f"Tool `{name}` error: {e}"
                })
        else:
            # Max iterations reached - ask for final answer
            logger.warning(f"[Task {log_id}] Max iterations reached, requesting final answer")
            state.messages.append({
                "role": "user",
                "content": "You have reached the maximum number of iterations. Please provide your final answer now based on what you have learned."
            })
            response = completion(
                messages=state.messages,
                model=DEFAULT_MODEL,
                temperature=0.0,
                top_p=0,
                seed=0,
            )
            assistant_content = response.choices[0].message.content or ""

        logger.info(f"[Task {log_id}] Completed task")
        if assistant_content:
            logger.debug(f"[Task {log_id}] Final response: {assistant_content}")
        else:
            logger.warning(f"[Task {log_id}] No final response produced")

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=assistant_content or "No response produced."))],
            name="Response",
        )

    async def _ensure_state(self, ctx_id: str, user_input: str) -> MCPContextState:
        """Ensure MCP connection state exists for this context."""
        mcp_url, _ = self._extract_mcp_url(user_input)
        if not mcp_url:
            raise ValueError("No MCP URL found in prompt")

        state = self.ctx_id_to_state.get(ctx_id)
        if state and state.url != mcp_url:
            await self._teardown_context(ctx_id)
            state = None

        if not state:
            client = MCPClient(mcp_url)
            await client.connect()

            tools_result = await client.list_tools()
            tools_desc = self._format_tools_description(tools_result.tools)
            tools_index = {tool.name for tool in tools_result.tools}

            messages = [{
                "role": "system",
                "content": f"{SYSTEM_PROMPT}\nAvailable MCP tools:\n{tools_desc}\n",
            }]

            state = MCPContextState(
                url=mcp_url,
                client=client,
                messages=messages,
                tools_index=tools_index,
                session_id=client.session_id,
            )
            self.ctx_id_to_state[ctx_id] = state

        return state

    async def _teardown_context(self, ctx_id: str) -> None:
        """Close and remove MCP connection for a context."""
        state = self.ctx_id_to_state.pop(ctx_id, None)
        if state:
            try:
                await state.client.close()
            except Exception as e:
                pass

    @staticmethod
    def _extract_mcp_url(text: str) -> tuple[Optional[str], Optional[str]]:
        """Extract MCP server URL and task ID from prompt text."""
        match = re.search(r'(?:MCP|mcp)[^:]*(?:at|available at)[:\s]+(\S+)', text)
        if match:
            url = match.group(1).strip()
            if not url.endswith('/mcp'):
                url = url.rstrip('/') + '/mcp'

            # Extract task ID from URL like .../tasks/{task_id}/mcp
            task_match = re.search(r'/tasks/([^/]+)/mcp', url)
            task_id = task_match.group(1) if task_match else None

            return url, task_id
        return None, None

    @staticmethod
    def _parse_action(response_text: str) -> Optional[dict[str, Any]]:
        """Parse single JSON action from LLM response."""
        json_str = None

        match = re.search(r'<json>\s*(.*?)\s*</json>', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if match:
                    json_str = match.group(1)

        parsed = json.loads(json_str if json_str else response_text)

        if isinstance(parsed, list):
            parsed = parsed[0] if parsed else None

        if isinstance(parsed, dict) and "name" in parsed:
            return parsed

        return None

    @staticmethod
    def _format_tools_description(tools) -> str:
        """Format tools list for inclusion in prompt."""
        serialized = [
            {
                "name": t.name,
                "description": getattr(t, "description", ""),
                "inputSchema": getattr(t, "inputSchema", {}),
            }
            for t in sorted(tools, key=lambda tool: tool.name or "")
        ]
        return json.dumps(serialized, indent=2, sort_keys=True)

    @staticmethod
    def _format_tool_result(result: Any) -> str:
        """Format tool result for inclusion in conversation."""
        if hasattr(result, "content"):
            try:
                return json.dumps(result.content, indent=2, ensure_ascii=False)
            except Exception:
                return str(result.content)
        try:
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception:
            return str(result)
