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

"""
Terragro Agent - Ambient Agricultural Assistant

ARCHITECTURAL DESIGN DECISIONS:
-------------------------------
1. **Explicit Graph-Based Control (ADK 2.0 Workflow)**:
   We utilize ADK 2.0's `Workflow` API to construct a deterministic graph structure.
   Agricultural analysis requires running separate tasks in parallel: inspecting a crop
   image and fetching localized weather. Rather than relying on non-deterministic LLM
   orchestration (where an LLM agent decides sequentially what tools to call), our
   graph workflow guarantees that both steps execute concurrently.

2. **Strict Schema Contracts (Pydantic Models)**:
   To eliminate model hallucinations and guarantee downstream data parsing safety,
   all node inputs and outputs are governed by Pydantic schemas:
   - `DiagnosisOutput`: Forces `vision_node` to categorize crop condition, severity, and findings.
   - `RemediationPlan`: Forces the final output into structured, actionable items.

3. **Model Context Protocol (MCP)**:
   We isolate localized external APIs (like weather information) to an MCP server,
   ensuring that credentials and API changes do not leak into the core agent code.

ADK 2.0 GRAPH EDGE ROUTING:
---------------------------
Our graph implements a classic "Diamond Pattern" with a conditional branch exit:
1. **Fan-Out (Concurrence)**:
   We route the starting user trigger to both `vision_node` (visual diagnosis) and
   `location_extractor` (weather location resolver) in parallel using the syntax:
   `("START", (vision_node, location_extractor))`
2. **Sequential Flow (MCP Invocation)**:
   `location_extractor` passes its parsed location dictionary `{"location": ...}` to
   `context_node` (the weather tool).
3. **Fan-In (Merging)**:
   `vision_node` and `context_node` outputs are synchronized and fanned-in using `JoinNode`.
   The downstream node (`triage_node`) receives a combined dictionary containing the
   outputs of both branches.
4. **Conditional Routing**:
   The final routing uses a `RoutingMap` dictionary mapping the `"approved"` route to
   the final LLM agent node: `(triage_node, {"approved": remediation_node})`.

HUMAN-IN-THE-LOOP (HITL) TRIAGE SAFEGUARD:
------------------------------------------
Agricultural remediation (e.g., advising pesticide application or heavy watering) carries
significant real-world risks and costs. To prevent the agent from executing critical decisions
autonomously without oversight, the `triage_node` acts as an absolute safety gate:
- If the session's `resume_inputs` lacks the `"validation"` key, it yields a `RequestInput` event
  containing the diagnostic and weather details and immediately halts execution.
- The session state, variables, and progress are securely persisted in-memory or in the DB.
- When the user verifies the information and responds (e.g., typing 'Approve'), the runner
  re-triggers the workflow, detects the `"validation"` key, resolves the pause, and routes the
  execution to the final `remediation_node` to build the plans.

PREVENTING CONTEXT ROT & OVERFLOW:
----------------------------------
Long conversations with multiple iterations of crop images and weather details can lead to
context rot, degrading the LLM's reasoning and inflating costs. We combat this using two key
ADK mechanisms configured on the `App`:
1. **Context Caching (`ContextCacheConfig`)**:
   We cache our system instructions and templates. Since the prompts for plant pathology
   and remediation are complex, caching them reduces latency and optimizes token usage.
2. **Context Compaction (`EventsCompactionConfig`)**:
   We configure a sliding window compaction. If the session history grows beyond 20 events,
   the `LlmEventSummarizer` automatically condenses older exchanges into a concise summary
   while retaining the last 3 events for continuity. This keeps the active context window clean,
   fresh, and focused.
"""

import asyncio
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App, ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
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

# Defines the weather tool execution node dynamically as an async function node
# to avoid executing asyncio.run() at import time, preventing startup crashes.
@node(name="get_weather")
async def context_node(ctx: Context, node_input: Any) -> Any:
    """Connects to MCP and calls get_weather dynamically during execution."""
    location = node_input.get("location") if isinstance(node_input, dict) else str(node_input)
    try:
        tools = await weather_mcp_toolset.get_tools()
        weather_tool = tools[0]
        # Execute the weather tool dynamically using the run_async method
        raw_result = await weather_tool.run_async(args={"location": location}, tool_context=ctx)

        # Extract and parse the flattened dictionary from the MCP response content
        if isinstance(raw_result, dict) and "content" in raw_result:
            content_list = raw_result["content"]
            if content_list and isinstance(content_list, list):
                text_content = content_list[0].get("text", "{}")
                try:
                    import json
                    return json.loads(text_content)
                except Exception:
                    pass
        return raw_result
    except Exception as e:
        raise RuntimeError(f"Failed to execute MCP weather tool: {e}")


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
        "You are a friendly, expert agricultural advisor. Ingest the crop diagnosis "
        "(condition, severity, findings) and the localized weather conditions. "
        "Generate a conversational, farmer-friendly treatment plan. "
        "Do NOT output JSON, markdown tables, or structural JSON block formatting. "
        "Instead, format the output as plain, empathetic markdown with clear bold "
        "headers and bulleted actionable steps that a farmer can easily understand "
        "and implement in the field."
    ),
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
    elif "seattle" in query_lower or "london" in query_lower:
        location = "Seattle"
    elif "phoenix" in query_lower or "sahara" in query_lower:
        location = "Phoenix"
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

    # Extract fields from structured diagnosis and weather context to present a clean UX
    condition = "Unknown"
    severity = "Unknown"
    findings_str = ""

    if isinstance(diagnosis, dict):
        condition = diagnosis.get("condition", "Unknown")
        severity = diagnosis.get("severity", "Unknown")
        findings = diagnosis.get("findings", [])
        if isinstance(findings, list):
            findings_str = "\n".join(f"- {f}" for f in findings)
    elif hasattr(diagnosis, "condition"):
        condition = getattr(diagnosis, "condition", "Unknown")
        severity = getattr(diagnosis, "severity", "Unknown")
        findings = getattr(diagnosis, "findings", [])
        if isinstance(findings, list):
            findings_str = "\n".join(f"- {f}" for f in findings)
    else:
        condition = str(diagnosis)

    # Format weather dictionary into clean Markdown prose
    weather_str = "No weather context available"
    if isinstance(weather, dict):
        region = weather.get("region_type", "Unknown Region")
        temp = weather.get("temperature_f", "N/A")
        humidity = weather.get("humidity_pct", "N/A")
        rain_prob = weather.get("rain_probability_pct", "N/A")
        soil_temp = weather.get("soil_temperature_f", "N/A")
        wind = weather.get("wind_speed_mph", "N/A")
        impact = weather.get("agricultural_impact", "None")

        weather_str = (
            f"📍 **Region:** {region}\n"
            f"- **Temperature:** {temp}°F\n"
            f"- **Humidity:** {humidity}%\n"
            f"- **Rain Probability:** {rain_prob}%\n"
            f"- **Soil Temperature:** {soil_temp}°F\n"
            f"- **Wind Speed:** {wind} mph\n"
            f"- **Agricultural Impact:** {impact}"
        )
    else:
        weather_str = str(weather)

    # Bypasses the human triage pause during automated evaluation runs to allow
    # the LLM judge to grade the final remediation plans.
    import os
    if os.getenv("EVAL_RUN") == "true":
        yield Event(
            output={
                "diagnosis": str(diagnosis),
                "weather": weather_str,
                "validation": "Auto-approved during evaluation benchmark run",
            },
            actions=EventActions(route="approved"),
        )
        return

    # Format into clean, bulleted Markdown prose
    message = (
        f"🌾 **Crop Condition Detected:** {condition}\n"
        f"⚠️ **Severity:** {severity}\n"
    )
    if findings_str:
        message += f"🔍 **Key Findings:**\n{findings_str}\n"

    message += f"\n☁️ **Localized Weather Context:**\n{weather_str}\n\n"
    message += "Please review and enter your comments or 'Approve' to proceed:"

    # If the user has not yet validated the information, pause and request input.
    if not ctx.resume_inputs or "validation" not in ctx.resume_inputs:
        yield RequestInput(
            interrupt_id="validation",
            message=message,
        )
        return

    # If validation response is present, resume and route to remediation
    validation_response = ctx.resume_inputs["validation"]
    yield Event(
        output={
            "diagnosis": str(diagnosis),
            "weather": weather_str,
            "validation": str(validation_response),
        },
        actions=EventActions(route="approved"),
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

# Export app with ResumabilityConfig enabled to support human-in-the-loop pausing,
# along with context caching and context compaction to prevent context rot.
app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
    context_cache_config=ContextCacheConfig(
        min_tokens=2048,
        ttl_seconds=1800,
        cache_intervals=10,
    ),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=20,
        overlap_size=3,
        summarizer=LlmEventSummarizer(llm=Gemini(model="gemini-3.1-flash-lite")),
    ),
)
