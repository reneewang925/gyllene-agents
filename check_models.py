import os
import requests
from dotenv import load_dotenv

# 加载 .env 文件中的 API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("未找到 API Key，请检查 .env 文件。")
else:
    # 向 Google 服务器请求模型列表
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        print("✅ 你的 API Key 支持以下 Gemini 模型：\n")
        models = response.json().get('models', [])
        for m in models:
            name = m['name'].replace('models/', '')
            if 'gemini' in name:
                print(f"- {name}")
    else:
        print(f"❌ 查询失败: {response.text}")