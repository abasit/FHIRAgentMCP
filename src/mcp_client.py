"""
MCP client wrapper for connecting to MCP servers.
"""

import logging
from contextlib import AsyncExitStack
from typing import Optional

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger("mcp_purple_agent")


class MCPClient:
    """MCP client with explicit lifecycle control."""

    def __init__(self, url: str):
        self.url = url
        self._stack: Optional[AsyncExitStack] = None
        self.session: Optional[ClientSession] = None

    async def connect(self) -> ClientSession:
        """Connect to the MCP server and initialize session."""
        logger.debug(f"Connecting to MCP server at {self.url}")
        self._stack = AsyncExitStack()

        try:
            read, write, _ = await self._stack.enter_async_context(
                streamable_http_client(self.url)
            )
            self.session = await self._stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            logger.debug(f"Connected to MCP server")
            return self.session

        except BaseExceptionGroup as eg:
            await self.close()
            for e in eg.exceptions:
                if isinstance(e, httpx.HTTPStatusError):
                    raise ConnectionError(
                        f"Could not connect to MCP server at {self.url}: "
                        f"{e.response.status_code} {e.response.reason_phrase}"
                    ) from e
            raise
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        """Close the MCP connection and cleanup resources."""
        if self._stack:
            try:
                await self._stack.aclose()
            except Exception:
                pass
        self._stack = None
        self.session = None

    async def list_tools(self):
        """List available tools from the MCP server."""
        if not self.session:
            raise RuntimeError("MCPClient not connected")
        return await self.session.list_tools()

    async def call_tool(self, tool_name: str, kwargs: dict):
        """Call a tool on the MCP server."""
        if not self.session:
            raise RuntimeError("MCPClient not connected")
        return await self.session.call_tool(tool_name, kwargs)