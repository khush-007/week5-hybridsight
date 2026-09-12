"""
Utility functions for processing images.
"""

import base64


def image_to_data_uri(filepath: str) -> str:
    """
    Convert a local image file into a base64 data URI.
    """

    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"