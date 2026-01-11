import argparse
import logging

import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor

# Configure logging
logging.basicConfig(level=logging.WARNING)

logging.getLogger("mcp_purple_agent").setLevel(logging.DEBUG)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description="Run the MCP-based purple agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9002, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="mcp_task_fulfillment",
        name="MCP Task Fulfillment",
        description="Handles user requests and completes tasks using FHIR MCP tools",
        tags=["mcp", "fhir", "medical"],
        examples=[]
    )

    agent_card = AgentCard(
        name="MCP Based FHIR Purple Agent",
        description="Agent that answers medical questions using FHIR data via MCP",
        url=args.card_url or f"http://{args.host}:{args.port}/",
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

    logger = logging.getLogger("mcp_purple_agent")
    logger.info(f"Starting MCP purple agent at {args.host}:{args.port}")

    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
