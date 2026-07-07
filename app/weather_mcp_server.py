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
def get_weather(location: str) -> dict:
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
        return {
            "region_type": "Rainy/Humid Region",
            "temperature_f": 54,
            "humidity_pct": 92,
            "rain_probability_pct": 85,
            "soil_temperature_f": 52,
            "wind_speed_mph": 10,
            "agricultural_impact": "High risk of foliar fungal pathogens (e.g., Late Blight, Downy Mildew)."
        }

    # 2. Arid / Extreme Heat Regions (e.g., Phoenix, Sahara)
    elif any(city in loc_lower for city in ["phoenix", "sahara", "desert", "arid"]):
        return {
            "region_type": "Arid/Desert Region",
            "temperature_f": 104,
            "humidity_pct": 12,
            "rain_probability_pct": 0,
            "soil_temperature_f": 89,
            "wind_speed_mph": 15,
            "agricultural_impact": "High risk of heat stress, drought wilting, and spider mite proliferation."
        }

    # 3. Coastal California (e.g., Salinas Valley)
    elif any(city in loc_lower for city in ["salinas", "california", "coastal"]):
        return {
            "region_type": "Salinas Valley",
            "temperature_f": 64,
            "humidity_pct": 85,
            "rain_probability_pct": 5,
            "soil_temperature_f": 59,
            "wind_speed_mph": 12,
            "agricultural_impact": "Moderate risk of powdery mildew due to persistent morning fog."
        }

    # 4. Midwest Corn Belt (e.g., Iowa)
    elif any(city in loc_lower for city in ["iowa", "midwest", "corn"]):
        return {
            "region_type": "Iowa Region",
            "temperature_f": 82,
            "humidity_pct": 65,
            "rain_probability_pct": 15,
            "soil_temperature_f": 71,
            "wind_speed_mph": 8,
            "agricultural_impact": "Excellent growth conditions; monitor for root rot if dampness increases."
        }

    # 5. Default General Weather
    else:
        return {
            "region_type": f"General Weather for {location}",
            "temperature_f": 72,
            "humidity_pct": 50,
            "rain_probability_pct": 20,
            "soil_temperature_f": 65,
            "wind_speed_mph": 5,
            "agricultural_impact": "Standard growing conditions."
        }

if __name__ == "__main__":
    mcp.run()
