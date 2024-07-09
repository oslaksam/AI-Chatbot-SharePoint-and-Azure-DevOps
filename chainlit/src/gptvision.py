import base64

import os

import langchain
from dotenv import dotenv_values, load_dotenv
from langchain_community.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage

langchain.debug = True


# Load environment variables from .env file
if os.path.exists(".env"):
    load_dotenv(override=True)
    config = dotenv_values(".env")

# Your Azure OpenAI GPT-4 Vision deployment endpoint
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "noendpoint")

# Your API key for authentication
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "nokey")

# Path to your image file
IMAGE_PATH = "../../architecture.jpg"

# Your prompt
PROMPT = "Describe this diagram and suggest improvements for better visual appeal."

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_openai_gpt4_vision(endpoint, api_key, image_data, prompt):
    """Send a request to the GPT-4 Vision API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "prompt": prompt,
        "image": image_data,  # Ensure this is correctly formatted per your deployment's requirements
    }

    chat = AzureChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024, deployment_name="gpt-4-vision")

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ]
            )
        ]
    )
    return msg.content

# Encode the image
encoded_image = encode_image(IMAGE_PATH)

# Call the API
result = call_openai_gpt4_vision(ENDPOINT, API_KEY, encoded_image, PROMPT)

print(result)
