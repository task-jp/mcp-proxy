"""Create an MCP server that proxies requests through an MCP client.

This server is created independent of any transport mechanism.
"""

import contextlib
import logging
import typing as t
from collections.abc import Awaitable, Callable

from mcp import server, types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)

RELOAD_MCP_TOOL = types.Tool(
    name="reload_mcp",
    description="Reload the backend MCP server process",
    inputSchema={"type": "object"},
)


class SessionManager:
    """Manage the lifecycle of a stdio MCP client session."""

    def __init__(self, params: StdioServerParameters) -> None:
        self._params = params
        self._stack: contextlib.AsyncExitStack | None = None

    async def start(self) -> ClientSession:
        """Start a new stdio client and session."""
        self._stack = contextlib.AsyncExitStack()
        stdio_streams = await self._stack.enter_async_context(stdio_client(self._params))
        return await self._stack.enter_async_context(ClientSession(*stdio_streams))

    async def reload(self) -> ClientSession:
        """Close the existing session and start a new one."""
        await self.close()
        return await self.start()

    async def close(self) -> None:
        """Clean up the current session resources."""
        if self._stack:
            await self._stack.aclose()
            self._stack = None


async def create_proxy_server(  # noqa: C901, PLR0915
    remote_app: ClientSession,
    *,
    reload_callback: Callable[[], Awaitable[ClientSession]] | None = None,
) -> server.Server[object]:
    """Create a server instance from a remote app."""
    logger.debug("Sending initialization request to remote MCP server...")
    response = await remote_app.initialize()
    capabilities = response.capabilities

    logger.debug("Configuring proxied MCP server...")
    app: server.Server[object] = server.Server(name=response.serverInfo.name)

    session_ref: list[ClientSession] = [remote_app]

    if capabilities.prompts:
        logger.debug("Capabilities: adding Prompts...")

        async def _list_prompts(_: t.Any) -> types.ServerResult:  # noqa: ANN401
            result = await session_ref[0].list_prompts()
            return types.ServerResult(result)

        app.request_handlers[types.ListPromptsRequest] = _list_prompts

        async def _get_prompt(req: types.GetPromptRequest) -> types.ServerResult:
            result = await session_ref[0].get_prompt(req.params.name, req.params.arguments)
            return types.ServerResult(result)

        app.request_handlers[types.GetPromptRequest] = _get_prompt

    if capabilities.resources:
        logger.debug("Capabilities: adding Resources...")

        async def _list_resources(_: t.Any) -> types.ServerResult:  # noqa: ANN401
            result = await session_ref[0].list_resources()
            return types.ServerResult(result)

        app.request_handlers[types.ListResourcesRequest] = _list_resources

        async def _list_resource_templates(_: t.Any) -> types.ServerResult:  # noqa: ANN401
            result = await session_ref[0].list_resource_templates()
            return types.ServerResult(result)

        app.request_handlers[types.ListResourceTemplatesRequest] = _list_resource_templates

        async def _read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
            result = await session_ref[0].read_resource(req.params.uri)
            return types.ServerResult(result)

        app.request_handlers[types.ReadResourceRequest] = _read_resource

    if capabilities.logging:
        logger.debug("Capabilities: adding Logging...")

        async def _set_logging_level(req: types.SetLevelRequest) -> types.ServerResult:
            await session_ref[0].set_logging_level(req.params.level)
            return types.ServerResult(types.EmptyResult())

        app.request_handlers[types.SetLevelRequest] = _set_logging_level

    if capabilities.resources:
        logger.debug("Capabilities: adding Resources...")

        async def _subscribe_resource(req: types.SubscribeRequest) -> types.ServerResult:
            await session_ref[0].subscribe_resource(req.params.uri)
            return types.ServerResult(types.EmptyResult())

        app.request_handlers[types.SubscribeRequest] = _subscribe_resource

        async def _unsubscribe_resource(req: types.UnsubscribeRequest) -> types.ServerResult:
            await session_ref[0].unsubscribe_resource(req.params.uri)
            return types.ServerResult(types.EmptyResult())

        app.request_handlers[types.UnsubscribeRequest] = _unsubscribe_resource

    if capabilities.tools or reload_callback:
        logger.debug("Capabilities: adding Tools...")
        has_backend_tools = bool(capabilities.tools)

        async def _list_tools(_: t.Any) -> types.ServerResult:  # noqa: ANN401
            tools_list: list[types.Tool] = []
            if has_backend_tools:
                result = await session_ref[0].list_tools()
                tools_list = list(result.tools)
            if reload_callback:
                tools_list.append(RELOAD_MCP_TOOL)
            return types.ServerResult(types.ListToolsResult(tools=tools_list))

        app.request_handlers[types.ListToolsRequest] = _list_tools

        async def _call_tool(req: types.CallToolRequest) -> types.ServerResult:
            if reload_callback and req.params.name == "reload_mcp":
                try:
                    new_session = await reload_callback()
                    await new_session.initialize()
                    session_ref[0] = new_session
                    return types.ServerResult(
                        types.CallToolResult(
                            content=[
                                types.TextContent(type="text", text="Reloaded successfully"),
                            ],
                        ),
                    )
                except Exception as e:  # noqa: BLE001
                    return types.ServerResult(
                        types.CallToolResult(
                            content=[types.TextContent(type="text", text=str(e))],
                            isError=True,
                        ),
                    )
            try:
                result = await session_ref[0].call_tool(
                    req.params.name,
                    (req.params.arguments or {}),
                )
                return types.ServerResult(result)
            except Exception as e:  # noqa: BLE001
                return types.ServerResult(
                    types.CallToolResult(
                        content=[types.TextContent(type="text", text=str(e))],
                        isError=True,
                    ),
                )

        app.request_handlers[types.CallToolRequest] = _call_tool

    async def _send_progress_notification(req: types.ProgressNotification) -> None:
        await session_ref[0].send_progress_notification(
            req.params.progressToken,
            req.params.progress,
            req.params.total,
        )

    app.notification_handlers[types.ProgressNotification] = _send_progress_notification

    async def _complete(req: types.CompleteRequest) -> types.ServerResult:
        result = await session_ref[0].complete(
            req.params.ref,
            req.params.argument.model_dump(),
        )
        return types.ServerResult(result)

    app.request_handlers[types.CompleteRequest] = _complete

    return app
