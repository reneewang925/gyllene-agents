import os
from pathlib import Path
from google import genai
from google.genai import types

def render_zen_image(prompt: str, output_path: str, aspect_ratio: str = "1:1", image_size: str = "4K") -> bool:
    """调用 Nano Banana Pro 生成单张图片并保存"""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size
            )
        )
    )

    for part in response.candidates[0].content.parts:
        if hasattr(part, "image") and part.image:
            img = part.as_image()
            img.save(str(output_file))
            return True
    return False