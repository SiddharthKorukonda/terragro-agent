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

from fastmcp import FastMCP

mcp = FastMCP("WeatherService")

@mcp.tool()
def get_weather(location: str) -> str:
    """Get the weather and atmospheric context for a specific location.

    Args:
        location: The location or region to query weather for (e.g., 'Iowa', 'Salinas Valley').
    """
    loc_lower = location.lower()
    if "iowa" in loc_lower or "midwest" in loc_lower:
        return (
            "Current Weather for Iowa Region:\n"
            "- Temperature: 82°F (Warm)\n"
            "- Humidity: 65% (Moderate-High)\n"
            "- Rain Probability: 15% (Dry conditions)\n"
            "- Soil Temperature: 71°F (Favorable)\n"
            "- Wind: 8 mph SE"
        )
    elif "salinas" in loc_lower or "coastal california" in loc_lower:
        return (
            "Current Weather for Salinas Valley:\n"
            "- Temperature: 64°F (Mild, overcast start)\n"
            "- Humidity: 85% (High coastal dampness)\n"
            "- Rain Probability: 5% (Mostly dry)\n"
            "- Soil Temperature: 59°F (Cooler)\n"
            "- Wind: 12 mph NW (Foggy coastal breeze)"
        )
    else:
        return (
            f"General Weather for {location}:\n"
            "- Temperature: 72°F\n"
            "- Humidity: 50%\n"
            "- Rain Probability: 20%\n"
            "- Wind: 5 mph"
        )

if __name__ == "__main__":
    mcp.run()
