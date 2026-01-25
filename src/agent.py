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

from mcp_client import MCPClient


warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
logging.getLogger("mcp.client.streamable_http").setLevel(logging.CRITICAL)

logger = logging.getLogger("mcp_purple_agent")

MAX_ITERATIONS = 10
DEFAULT_MODEL = "openai/gpt-4o-mini"

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


class MCPServerError(Exception):
    """Raised when MCP server fails during tool operations."""
    pass


class Agent:
    """Purple agent that uses MCP tools to answer questions."""

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Process message and respond using MCP tools."""
        user_input = get_message_text(message)
        mcp_url, task_id = self._extract_mcp_url(user_input)
        log_id = task_id or message.context_id

        logger.info(f"[Task {log_id}] Received task")

        client = None
        try:
            if not mcp_url:
                raise ValueError("No MCP URL found in prompt")

            client = MCPClient(mcp_url)
            await client.connect()
            logger.debug(f"[Task {log_id}] Connected to MCP at {mcp_url}")

            result = await self._run_agent_loop(client, user_input, log_id)

        except ConnectionError as e:
            logger.error(f"[Task {log_id}] MCP connection error: {e}")
            result = f"Failed to complete task: {e}"
        except MCPServerError as e:
            logger.error(f"[Task {log_id}] MCP server error: {e}")
            result = "Failed to complete task: MCP server error"
        except Exception as e:
            logger.exception(f"[Task {log_id}] Error: {e}")
            result = "Failed to complete task: Internal error."
        finally:
            if client:
                await client.close()

        logger.info(f"[Task {log_id}] Completed task")

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=result or "No response produced."))],
            name="Response",
        )

    async def _run_agent_loop(self, client: MCPClient, user_input: str, log_id: str) -> str:
        """Run the agent loop: LLM -> tool calls -> repeat until done."""
        # Get available tools
        try:
            tools_result = await client.list_tools()
        except Exception as e:
            raise MCPServerError(f"list_tools failed: {e}") from e

        tools_desc = self._format_tools_description(tools_result.tools)
        tools_index = {tool.name for tool in tools_result.tools}

        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\nAvailable MCP tools:\n{tools_desc}\n"},
            {"role": "user", "content": user_input},
        ]

        for i in range(MAX_ITERATIONS):
            logger.debug(f"[Task {log_id}] Iteration {i + 1}/{MAX_ITERATIONS}")

            # Call LLM
            try:
                response = completion(
                    messages=messages,
                    model=DEFAULT_MODEL,
                    temperature=0.0,
                    top_p=0,
                    seed=0,
                )
            except Exception as e:
                logger.error(f"[Task {log_id}] LLM error: {e}")
                return "Failed to complete task: Internal error."

            assistant_content = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant_content})
            logger.debug(f"[Task {log_id}] LLM response:\n{assistant_content}")

            # Parse action
            try:
                action = self._parse_action(assistant_content)
                logger.debug(f"[Task {log_id}] Action: {action.get('name') if action else None}")
            except Exception as e:
                logger.warning(f"[Task {log_id}] Failed to parse response: {e}")
                return "Failed to complete task: Internal error."

            if not action:
                logger.warning(f"[Task {log_id}] No action found in response")
                return "Failed to complete task: Internal error."

            name = action.get("name")
            kwargs = action.get("kwargs", {})

            # Final response
            if name == "response":
                logger.debug(f"[Task {log_id}] Got final response")
                return assistant_content

            # Unknown tool
            if name not in tools_index:
                logger.warning(f"[Task {log_id}] Unknown tool: {name}")
                messages.append({
                    "role": "user",
                    "content": f"Error: Unknown tool '{name}'. Available tools: {sorted(tools_index)}"
                })
                continue

            # Call tool
            logger.debug(f"[Task {log_id}] Calling tool: {name}, args: {kwargs}")
            try:
                result = await client.call_tool(name, kwargs)
                result_text = self._format_tool_result(result)
                logger.debug(f"[Task {log_id}] Tool result:\n{result_text}")
                messages.append({
                    "role": "user",
                    "content": f"Tool `{name}` result:\n{result_text}"
                })
            except Exception as e:
                raise MCPServerError(f"call_tool '{name}' failed: {e}") from e

        # Max iterations - force final answer
        logger.warning(f"[Task {log_id}] Max iterations reached, requesting final answer")
        messages.append({
            "role": "user",
            "content": "You have reached the maximum number of iterations. Please provide your final answer now."
        })

        try:
            response = completion(
                messages=messages,
                model=DEFAULT_MODEL,
                temperature=0.0,
                top_p=0,
                seed=0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"[Task {log_id}] LLM error on final answer: {e}")
            return "Failed to complete task: Internal error."

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
