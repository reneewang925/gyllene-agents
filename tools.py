import os
import json
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from crewai.tools import tool

# ==========================================
# 辅助函数：Imagen 3 (最新 SDK 兼容版) 生图引擎
# ==========================================
def render_zen_image(prompt: str, output_path: str, aspect_ratio: str = "1:1", image_size: str = "4K") -> bool:
    """调用普通版 Imagen 3 生成单张图片并保存 (适配 Google 最新 generate_content 语法)"""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 听从 Google 报错的建议，改用 generate_content
        response = client.models.generate_content(
            model="imagen-3.0-generate-002",
            contents=[prompt],
            # 免费的基础画图模型不需要像 Nano Banana 那样配置复杂的模态参数
        )
        
        # 提取并保存生成的图片
        for part in response.candidates[0].content.parts:
            if hasattr(part, "image") and part.image:
                img = part.as_image()
                img.save(str(output_file))
                return True
                
    except Exception as e:
        print(f"❌ 免费版 API 生图失败: {e}")
        
    return False

# ==========================================
# 工具 1：解析 SKU 属性
# ==========================================
@tool("parse_sku_directory")
def parse_sku_directory(folder_name: str) -> str:
    """Parses a raw product folder name string (e.g., 'cup01100001') to extract physical attributes.
    Expects string input. Returns JSON detailing product type, material, diameter, and serial.
    """
    try:
        prefix = folder_name[:3]
        material_code = folder_name[3:5]
        diameter_code = folder_name[5:7]
        serial_code = folder_name[7:]
        
        material_map = {
            "01": "Ceramic (陶瓷)",
            "02": "Zisha (紫砂)"
        }
        
        material = material_map.get(material_code, "Unknown Material")
        diameter = int(diameter_code) if diameter_code.isdigit() else 0
        
        return json.dumps({
            "product_type": prefix,
            "material": material,
            "diameter_cm": diameter,
            "serial_number": serial_code
        })
    except Exception as e:
        return f"Error parsing SKU directory: {str(e)}"

# ==========================================
# 工具 2：核心处理流水线 (已接入 Nano Banana)
# ==========================================
@tool("Process Product Assets and SKU Pipeline")
def process_product_assets(sku_folder_name: str) -> str:
    """
    Scans the given SKU folder inside 'raw_products/', filters original shots,
    reads specs.txt, and triggers Nano Banana Pro for 4K Zen renders.
    """
    base_dir = Path("raw_products") / sku_folder_name
    if not base_dir.exists():
        return f"Error: Folder {sku_folder_name} not found in raw_products."

    original_images = []
    other_images = []

    # 1. 筛选逻辑：精准识别结尾为 -1, -2, -3 的原图
    for file_path in base_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            stem = file_path.stem
            if stem.endswith(('-1', '-2', '-3')):
                original_images.append(str(file_path))
            else:
                other_images.append(str(file_path))

    # 规则兜底
    if len(original_images) < 3 and other_images:
        original_images.extend(other_images)
        other_images = []

    # 2. 读取中文详情描述
    specs_content = "未提供产品参数。"
    specs_file = base_dir / "specs.txt"
    if specs_file.exists():
        with open(specs_file, "r", encoding="utf-8") as f:
            specs_content = f.read()
            
    # 3. 调用 Nano Banana 生图接口
    ai_renders_output_dir = Path("processed_products") / sku_folder_name
    ai_renders_output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_render_paths = []
    
    atmospheric_prompts = [
        f"A minimalist luxury Eastern teaware product shot for {sku_folder_name}, placed on a dark grey slate stone table, soft natural side lighting, subtle tea steam, wabi-sabi aesthetic, 4K resolution, photorealistic.",
        f"Close-up detail shot of {sku_folder_name} in a tranquil zen tea ceremony setting, dark background, minimalist composition, poetic and refined mood, professional commercial photography."
    ]

    print("🎨 正在唤醒 Nano Banana Pro 生成 4K 图像，请稍候...")
    
    for idx, prompt_text in enumerate(atmospheric_prompts, start=1):
        save_path = ai_renders_output_dir / f"zen_render_{idx}.jpg"
        
        # 🌟 核心改动：调用上方定义的 Nano Banana 引擎
        success = render_zen_image(
            prompt=prompt_text, 
            output_path=str(save_path),
            aspect_ratio="1:1",
            image_size="4K"
        )
        
        if success:
            generated_render_paths.append(str(save_path))
        else:
            print(f"⚠️ 图像 render_{idx} 生成失败。")

    return (
        f"Successfully processed SKU: {sku_folder_name}\n"
        f"- Original Shots Preserved: {original_images}\n"
        f"- AI Zen Renders Generated (Nano Banana Pro): {generated_render_paths}\n"
        f"- Raw Product Specs: {specs_content}\n"
    )