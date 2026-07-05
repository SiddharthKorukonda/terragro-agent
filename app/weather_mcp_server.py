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
Weather MCP Server

ARCHITECTURAL DESIGN DECISION:
------------------------------
We utilize the Model Context Protocol (MCP) as a decoupling mechanism to separate
highly localized and external data queries (such as weather & environmental metrics)
from the core agent logic. By running a local MCP server that communicates via a
standardized JSON-RPC protocol over stdio, we ensure that:
1. External integrations can be hot-swapped, updated, or mocked independently of the agent.
2. The agent communicates with the toolset via a standard interface, keeping it lightweight.
3. Network calls are quarantined, minimizing side-effects inside the agent's graph engine.

PREVENTING CONTEXT ROT & REDUCING OVERHEAD:
------------------------------------------
In long-running agricultural loops, raw API responses with redundant JSON metadata
(e.g., coordinate listings, timestamps, raw wind vectors) can pollute the LLM's
context window, leading to context rot, high costs, and attention drift.
This MCP tool combats context rot by:
1. Filtering and cleaning raw inputs before returning them.
2. Formating response payloads into concise, structured key-value strings that provide
   only the agricultural metrics (temp, humidity, rain probability, soil temp)
   relevant to plant pathology.
"""

from fastmcp import FastMCP

# Instantiate FastMCP server to expose localized weather tools
mcp = FastMCP("WeatherService")

@mcp.tool()
def get_weather(location: str) -> str:
    """Get the weather and atmospheric context for a specific location.

    This tool is used to retrieve temperature, humidity, rain probability,
    and soil conditions which are crucial for diagnosing plant diseases
    and recommending crop remediation plans.

    Args:
        location: The location or region to query weather for (e.g. 'Seattle', 'Phoenix').
    """
    loc_lower = location.lower()

    # 1. Rainy / High Humidity Regions (e.g., Seattle, London)
    if any(city in loc_lower for city in ["seattle", "london", "rainy"]):
        return (
            "Current Weather for Rainy/Humid Region:\n"
            "- Temperature: 54°F (Cool)\n"
            "- Humidity: 92% (Very High - Fog & Drizzle)\n"
            "- Rain Probability: 85% (Continuous wet conditions)\n"
            "- Soil Temperature: 52°F (Cold, saturated soil)\n"
            "- Wind: 10 mph SW\n"
            "- AGRICULTURAL IMPACT: High risk of foliar fungal pathogens (e.g., Late Blight, Downy Mildew)."
        )

    # 2. Arid / Extreme Heat Regions (e.g., Phoenix, Sahara)
    elif any(city in loc_lower for city in ["phoenix", "sahara", "desert", "arid"]):
        return (
            "Current Weather for Arid/Desert Region:\n"
            "- Temperature: 104°F (Extreme Heat)\n"
            "- Humidity: 12% (Extremely Low)\n"
            "- Rain Probability: 0% (Drought conditions)\n"
            "- Soil Temperature: 89°F (Very Hot, dry topsoil)\n"
            "- Wind: 15 mph E\n"
            "- AGRICULTURAL IMPACT: High risk of heat stress, drought wilting, and spider mite proliferation."
        )

    # 3. Coastal California (e.g., Salinas Valley)
    elif any(city in loc_lower for city in ["salinas", "california", "coastal"]):
        return (
            "Current Weather for Salinas Valley:\n"
            "- Temperature: 64°F (Mild, overcast start)\n"
            "- Humidity: 85% (High coastal dampness)\n"
            "- Rain Probability: 5% (Mostly dry)\n"
            "- Soil Temperature: 59°F (Cooler)\n"
            "- Wind: 12 mph NW (Foggy coastal breeze)\n"
            "- AGRICULTURAL IMPACT: Moderate risk of powdery mildew due to persistent morning fog."
        )

    # 4. Midwest Corn Belt (e.g., Iowa)
    elif any(city in loc_lower for city in ["iowa", "midwest", "corn"]):
        return (
            "Current Weather for Iowa Region:\n"
            "- Temperature: 82°F (Warm)\n"
            "- Humidity: 65% (Moderate-High)\n"
            "- Rain Probability: 15% (Dry conditions)\n"
            "- Soil Temperature: 71°F (Favorable)\n"
            "- Wind: 8 mph SE\n"
            "- AGRICULTURAL IMPACT: Excellent growth conditions; monitor for root rot if dampness increases."
        )

    # 5. Default General Weather
    else:
        return (
            f"General Weather for {location}:\n"
            "- Temperature: 72°F (Mild)\n"
            "- Humidity: 50% (Moderate)\n"
            "- Rain Probability: 20% (Low chance of light rain)\n"
            "- Soil Temperature: 65°F (Normal)\n"
            "- Wind: 5 mph\n"
            "- AGRICULTURAL IMPACT: Standard growing conditions."
        )

if __name__ == "__main__":
    mcp.run()
