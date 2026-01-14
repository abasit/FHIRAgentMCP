# FHIR Purple Agent (MCP)

Baseline purple agent for FHIR Agent Evaluator benchmark. Uses A2A and MCP protocols.
Built using the template https://github.com/RDI-Foundation/agent-template

## Running
```bash
# Install dependencies
uv sync

# Configure environment
cp sample.env .env
# Edit .env with your OpenAI API key

# Run the server
uv run src/server.py --port 9009
```

## Docker
```bash
docker build -t fhir-purple-agent-mcp .
docker run --env-file .env -p 9009:9009 fhir-purple-agent-mcp
```