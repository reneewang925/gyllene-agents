import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

# 🌟 修复：导入当前 tools.py 中真实存在的两个工具
from tools import process_product_assets, parse_sku_directory

def main():
    load_dotenv()

    # 1. 强制使用 CrewAI 官方的 LLM 类，精准锁定模型
    gemini_llm = LLM(
        model="gemini/gemini-3.5-flash-lite",  # 如果报错 503，这里依然可以切回 flash
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7
    )

    # 2. 装配 1 号位产品运营员工
    merchandiser_agent = Agent(
        role="Senior Luxury E-commerce Merchandiser",
        goal="Extract product specs and write elegant, zen-inspired luxury product descriptions.",
        backstory="You are a veteran merchandiser for a luxury Eastern teaware brand. You appreciate wabi-sabi aesthetics and precise craftsmanship.",
        tools=[process_product_assets, parse_sku_directory],  # 👈 挂载正确的工具
        llm=gemini_llm,
        verbose=True
    )   

    # 3. 定义任务：解析 SKU -> 处理图片与中文翻译 -> 撰写英文文案
    draft_pdp_task = Task(
        description=(
            "1. Use the `parse_sku_directory` tool to understand the SKU 'cup01100001'.\n"
            "2. Use the `process_product_assets` tool to analyze the folder 'cup01100001' and trigger Nano Banana 4K image generation.\n"
            "3. Read the 'Raw Product Specs' (which is in Chinese) returned by the tool.\n"
            "4. Translate the Chinese specifications into English.\n"
            "5. Polish the translation into a high-end, Zen-inspired luxury product description.\n"
            "6. Structure the final copy using three distinct headings: 'The Object', 'The Artisan', and 'The Ritual'."
        ),
        expected_output="A beautifully formatted Markdown product description in English.",
        agent=merchandiser_agent
    )
    
    # 4. 组装并启动
    crew = Crew(
        agents=[merchandiser_agent],
        tasks=[draft_pdp_task],
        process=Process.sequential,
        verbose=True
    )

    # 确保测试目录存在
    os.makedirs("raw_products/cup01100001", exist_ok=True)
    os.makedirs("processed_products", exist_ok=True)
    
    # 5. 执行工作流
    print("Kicking off the CrewAI workflow...")
    result = crew.kickoff()
    
    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    print(result)

if __name__ == "__main__":
    main()