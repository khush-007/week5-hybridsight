from langchain_core.tools import tool
from image_utils import image_to_data_uri
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


@tool
def describe_image(image_path: str, question: str) -> str:
    """
    Analyze an uploaded image and answer the user's question
    about that image.

    image_path: Local path of the uploaded image.
    question: User's question about the image.
    """

    try:
        image_data = image_to_data_uri(image_path)

        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Could not process the image: {e}"