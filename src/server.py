"""
FHIR Purple Agent Server (MCP Mode)

A2A-compatible agent that answers medical questions using FHIR data.
Connects to green agent's MCP server to access tools.

Usage:
    python server.py --host 0.0.0.0 --port 9009
"""

import argparse
import logging

import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor

logging_level = logging.DEBUG

logging.basicConfig(level=logging.WARNING)
logging.getLogger("mcp_purple_agent").setLevel(logging_level)

logger = logging.getLogger("mcp_purple_agent")


def main():
    parser = argparse.ArgumentParser(description="Run the FHIR purple agent (MCP mode).")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="Public URL for agent card")
    args = parser.parse_args()

    base_url = args.card_url or f"http://{args.host}:{args.port}"

    skill = AgentSkill(
        id="fhir_task_fulfillment",
        name="FHIR Task Fulfillment",
        description="Answers medical questions and performs clinical tasks using FHIR data via MCP",
        tags=["mcp", "fhir", "medical", "healthcare"],
        examples=[],
    )

    agent_card = AgentCard(
        name="FHIR Purple Agent (MCP)",
        description="Agent that answers medical questions using FHIR data via MCP tools",
        url=base_url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    logger.info(f"Starting purple agent at {base_url}")
    uvicorn.run(server.build(), host=args.host, port=args.port, log_level="warning")


if __name__ == '__main__':
    main()