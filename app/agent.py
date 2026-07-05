# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.models import Gemini
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.workflow import JoinNode, Workflow, node
from google.genai import types
from mcp import StdioServerParameters
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# 1. Models & Pydantic Schemas
# -----------------------------------------------------------------------------

# Output schema for crop diagnosis (vision_node)
class DiagnosisOutput(BaseModel):
    condition: str = Field(
        description="The diagnosed crop condition or disease (e.g. Healthy, Late Blight, Aphid Infestation)"
    )
    severity: str = Field(
        description="The severity level: Low, Medium, or High"
    )
    findings: list[str] = Field(
        description="Key visual symptoms identified in the crop image"
    )


# Output schema for final remediation plan (remediation_node)
class RemediationPlan(BaseModel):
    diagnosis: str = Field(
        description="Summary of the validated crop diagnosis"
    )
    weather_context: str = Field(
        description="Summary of the weather context used"
    )
    remediation_steps: list[str] = Field(
        description="Actionable remediation steps and recommendations"
    )
    preventative_measures: list[str] = Field(
        description="Preventative measures for future crop protection"
    )


# -----------------------------------------------------------------------------
# 2. Weather MCP Toolset Connection & Tool Node Extraction
# -----------------------------------------------------------------------------

weather_mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uv",
            args=["run", "python", "app/weather_mcp_server.py"],
        )
    )
)

# Fetch the local weather tool synchronously at import time so we can place
# it directly in the Workflow edge layout.
try:
    tools = asyncio.run(weather_mcp_toolset.get_tools())
    context_node = tools[0]
except Exception as e:
    raise RuntimeError(f"Failed to load MCP weather tool: {e}")


# -----------------------------------------------------------------------------
# 3. Declarative LLM Agent Nodes
# -----------------------------------------------------------------------------

# Diagnoses crop conditions from a multimodal image
vision_node = LlmAgent(
    name="vision_node",
    model=Gemini(
        model="gemini-3.1-flash-lite",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an expert plant pathologist. Analyze the provided image of "
        "the crop and diagnose its condition. Identify any specific diseases, "
        "pests, or nutrient deficiencies."
    ),
    output_schema=DiagnosisOutput,
    output_key="diagnosis",
)

# Generates the final remediation plan
remediation_node = LlmAgent(
    name="remediation_node",
    model=Gemini(
        model="gemini-3.1-flash-lite",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an agricultural expert. Generate a detailed, structured crop "
        "remediation plan based on the crop diagnosis, weather context, and user "
        "validation feedback."
    ),
    output_schema=RemediationPlan,
    output_key="remediation_plan",
)


# -----------------------------------------------------------------------------
# 4. Function Nodes (Location Extractor & Triage Node)
# -----------------------------------------------------------------------------

@node(name="location_extractor")
def location_extractor(node_input: Any) -> dict:
    """Extracts location text from the user input content."""
    query_text = ""
    if hasattr(node_input, "parts") and node_input.parts:
        query_text = "".join(part.text for part in node_input.parts if part.text)
    elif isinstance(node_input, dict):
        query_text = node_input.get("location", "") or node_input.get("query", "")
    elif isinstance(node_input, str):
        query_text = node_input

    # Basic weather matching logic
    location = "Iowa"
    query_lower = query_text.lower()
    if "salinas" in query_lower or "california" in query_lower:
        location = "Salinas Valley"
    elif "iowa" in query_lower or "midwest" in query_lower:
        location = "Iowa"
    elif query_text.strip():
        location = query_text.strip()

    return {"location": location}


@node(name="triage_node")
async def triage_node(ctx: Context, node_input: dict) -> Any:
    """Pauses execution via RequestInput for human validation of crop/weather context."""
    # node_input is the merged output from JoinNode:
    # {"vision_node": ..., "get_weather": ...}
    diagnosis = node_input.get("vision_node")
    weather = node_input.get("get_weather")

    # If the user has not yet validated the information, pause and request input.
    if not ctx.resume_inputs or "validation" not in ctx.resume_inputs:
        yield RequestInput(
            interrupt_id="validation",
            message=(
                f"=== Triage Validation Required ===\n"
                f"Crop Diagnosis: {diagnosis}\n"
                f"Localized Weather: {weather}\n\n"
                f"Please review and enter your comments or 'Approve' to proceed:"
            ),
        )
        return

    # If validation response is present, resume and route to remediation
    validation_response = ctx.resume_inputs["validation"]
    yield Event(
        output={
            "diagnosis": str(diagnosis),
            "weather": str(weather),
            "validation": str(validation_response),
        },
        route="approved",
    )


# -----------------------------------------------------------------------------
# 5. Workflow and App Scaffolding
# -----------------------------------------------------------------------------

join_node = JoinNode(name="join_node")

# Define Workflow graph layout (edges)
root_agent = Workflow(
    name="terragro_agent_workflow",
    edges=[
        ("START", (vision_node, location_extractor)),
        (location_extractor, context_node),
        ((vision_node, context_node), join_node),
        (join_node, triage_node),
        (triage_node, {"approved": remediation_node}),
    ],
    description="Ambient agricultural assistant with crop diagnosis, localized weather context, and human triage.",
)

# Export app with ResumabilityConfig enabled to support human-in-the-loop pausing
app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
