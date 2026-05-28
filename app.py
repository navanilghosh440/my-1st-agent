from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

@tool
def my_custom_tool(city: str) -> str:
    """A comprehensive meteorological tool that dynamically looks up coordinates for ANY city worldwide, 
    fetches its live UV Index, and provides precise sun safety and skin protection advice.
    Args:
        city: The name of any global town or city (e.g., 'Kolkata', 'Paris', 'Los Angeles', 'Nairobi').
    """
    try:
        # Step 1: Dynamic Geocoding (Translate city name to latitude and longitude)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_response = requests.get(geo_url).json()
        geo_results = geo_response.get("results", [])
        
        if not geo_results:
            return f"The Weatherman Radar could not locate '{city}'. Please check the spelling and try again."
            
        # Extract location specifics from the top search result
        location_data = geo_results[0]
        lat = location_data.get("latitude")
        lon = location_data.get("longitude")
        country = location_data.get("country", "Unknown Country")
        resolved_name = location_data.get("name", city.title())

        # Step 2: Fetch Live UV Metrics using the freshly acquired coordinates
        uv_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=uv_index,is_day"
        uv_response = requests.get(uv_url).json()
        current_data = uv_response.get("current", {})
        
        uv_index = current_data.get("uv_index", 0.0)
        is_day = current_data.get("is_day", 1)
        
        # If it's night time at the location, UV exposure is non-existent
        if is_day == 0:
            return f"☀️ Live Sun Safety Report for {resolved_name} ({country}):\n- It is currently night time there. The UV Index is 0.0.\n- Action Required: No sun protection needed!"
            
        # Step 3: Mathematical risk categorization based on World Health Organization metrics
        if uv_index <= 2.0:
            risk = "Low Risk"
            protection = "Safe to stay outside! Minimal protection needed. Wear sunglasses if it's exceptionally bright."
        elif uv_index <= 5.0:
            risk = "Moderate Risk"
            protection = "Sun protection required. Apply SPF 15+ sunscreen, wear a hat, and seek shade during midday hours."
        elif uv_index <= 7.0:
            risk = "High Risk"
            protection = "Protection essential! Apply SPF 30+ sunscreen every 2 hours. Reduce time in direct sunlight between 11 AM and 4 PM."
        elif uv_index <= 10.0:
            risk = "Very High Risk"
            protection = "Dangerous conditions. SPF 50+ sunscreen is mandatory. Wear protective clothing, a wide-brimmed hat, and avoid direct exposure."
        else:
            risk = "Extreme Risk"
            protection = "Extreme hazard! Skin can burn in minutes. Stay indoors if possible. If outside, maximize shade and use maximum SPF protection."
            
        return f"☀️ Live Sun Safety Report for {resolved_name} ({country}):\n- Coordinates: Lat {lat}, Lon {lon}\n- Current UV Index: {uv_index} ({risk})\n- Action Required: {protection}"
        
    except Exception as e:
        return f"Unable to establish a connection with the meteorological network: {str(e)}"   

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer,DuckDuckGoSearchTool, my_custom_tool], ## add your tools here (don't remove final answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()