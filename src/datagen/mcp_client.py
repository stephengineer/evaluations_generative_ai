"""
MCP client for fetching schemas and business data from the SUT MCP server.

Uses the MCP Python SDK's StreamableHTTP transport to connect to a running
MCP server and call its tools.
"""

import json
import sys
from typing import Any, cast

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.common.logger import get_logger

logger = get_logger(__name__)


class McpDataClient:
    """Async context-manager that talks to the SUT MCP server over Streamable HTTP."""

    def __init__(self, server_url: str = "http://127.0.0.1:8000/mcp/"):
        self._url = server_url
        self._session: ClientSession | None = None
        self._transport_ctx: Any = None
        self._session_ctx: Any = None

    async def __aenter__(self) -> "McpDataClient":
        self._transport_ctx = streamable_http_client(self._url)
        try:
            read_stream, write_stream, _ = await self._transport_ctx.__aenter__()
        except BaseException:
            await self._transport_ctx.__aexit__(*sys.exc_info())
            raise

        session_entered = False
        try:
            self._session_ctx = ClientSession(read_stream, write_stream)
            self._session = await self._session_ctx.__aenter__()
            session_entered = True
            await self._session.initialize()
        except BaseException:
            if session_entered:
                await self._session_ctx.__aexit__(*sys.exc_info())
            await self._transport_ctx.__aexit__(*sys.exc_info())
            self._session_ctx = None
            self._session = None
            raise

        logger.info("Connected to MCP server at %s", self._url)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._session_ctx is not None:
            await self._session_ctx.__aexit__(exc_type, exc_val, exc_tb)
        if self._transport_ctx is not None:
            await self._transport_ctx.__aexit__(exc_type, exc_val, exc_tb)

    async def call_tool(self, tool_name: str) -> Any:
        """Call an MCP tool by name and return the parsed JSON result."""
        assert self._session is not None, "Client not connected"
        result = await self._session.call_tool(tool_name)
        if result.isError:
            raise RuntimeError(f"MCP tool '{tool_name}' returned error: {result}")
        text_parts = [block.text for block in result.content if hasattr(block, "text")]
        raw = "".join(text_parts)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    async def fetch_schema(self, schema_tool: str) -> dict[str, Any]:
        """Fetch a JSON schema from the MCP server."""
        logger.info("Fetching schema via tool '%s'", schema_tool)
        data = await self.call_tool(schema_tool)
        if isinstance(data, str):
            data = json.loads(data)
        return cast(dict[str, Any], data)

    async def fetch_business_data(self, tool_names: list[str]) -> dict[str, Any]:
        """Call multiple business-data tools and return results keyed by tool name."""
        results: dict[str, Any] = {}
        for name in tool_names:
            logger.info("Fetching business data via tool '%s'", name)
            results[name] = await self.call_tool(name)
        return results
