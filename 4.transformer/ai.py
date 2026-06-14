import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path
import torch
from diffusers import DiffusionPipeline

custom_cache_dir = Path(r"D:\HuggingfaceModels")

pipe = DiffusionPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-Z-Image",
    torch_dtype=torch.bfloat16,
    cache_dir=custom_cache_dir,
)

# 极致省显存、平抑功耗的防关机套装
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# ---------------------------------------------------------
# 【核心重构区 1】：使用字典集中管理所有的提示词
# 以后增加第7、第8个提示词，只需要往这里面加一行就行了
# ---------------------------------------------------------
prompts_dict = {
    1: "One China girl, highly detailed beautiful faces, clear facial features, best quality, uncovered chests, expose the nipples, smile, full-body photo, long legs, a complete body and limbs, sexy", 
    2: "One American white girl, highly detailed beautiful faces, clear facial features, best quality, uncovered chests, expose the nipples, smile, full-body photo, long legs, a complete body and limbs, sexy",
    3: "One China girl, highly detailed beautiful faces, clear facial features, best quality, uncovered chests, expose the nipples, smile, full-body photo, long legs, kneel, a complete body and limbs, sexy",
    4: "One American white girl, highly detailed beautiful faces, clear facial features, best quality, uncovered chests, expose the nipples, smile, full-body photo, long legs, kneel, a complete body and limbs, sexy",
    5: "One China girl, white skin, highly detailed beautiful faces, clear facial features, best quality, uncovered chests, expose the nipples, smile, full-body photo, long legs, kneel, devide legs, private can be seen, have pubes, Just the outline of the vulva, without any protrusions, a complete body and limbs, sexy, like a av film cover photo",
    6: "One American white girl, highly detailed beautiful faces, clear facial features, best quality, uncovered chests, expose the nipples, smile, full-body photo, long legs, kneel, devide legs, private can be seen, show vulva, a complete body and limbs, sexy"
}

my_negative_prompt = "faceless, bad face, blank face, deformed, ugly, mutated, poorly drawn face, bad anatomy, missing facial features, blurred, any clothes, cartoon, more than one person, black skin, scare"
# 【核心重构区 2】：控制中心 (你想怎么跑，在这里一键设置)
# 变量 A：选择你要运行的提示词编号 (列表形式)
# 如果想运行全部，就写：[1, 2, 3, 4, 5, 6]
# 如果今天只想测试第 1 和第 5 个，就写：[1, 5]
# selected_prompts = [1, 2, 3, 4, 5, 6] 
selected_prompts = [5] 

# 变量 B：每个提示词生成几张图片？
images_per_prompt = 1

# 变量 C：全局推理步数
global_steps = 100

def progress_callback(pipe, step_index, timestep, callback_kwargs):
    # 用 \r 可以让进度条在同一行刷新，不会刷满整个屏幕
    print(f"🔄 绘图进行中: 已完成 第 {step_index + 1} 步 / 共 {global_steps} 步", end='\r')
    return callback_kwargs

print("模型加载完毕，开始按照计划生成高画质图片...")

# 【核心重构区 3】：双重循环 (外层切提示词，内层生成图片)
# 外层循环：遍历你选中的提示词编号
for prompt_id in selected_prompts:
    current_prompt = prompts_dict[prompt_id]
    print(f"\n\n========================================================")
    print(f"🚀 开始处理提示词组别: 【第 {prompt_id} 组】")
    print(f"========================================================")

    # 内层循环：根据当前提示词，生成你指定数量的图片
    for i in range(1, images_per_prompt + 1):
        print(f"\n▶ 正在生成 组别 {prompt_id} 的 第 {i} 张图片 (当前组共需生成 {images_per_prompt} 张)")
        
        image = pipe(
            prompt=current_prompt,
            negative_prompt=my_negative_prompt,
            height=512,
            width=512,
            guidance_scale=5.0,
            num_inference_steps=global_steps,
            callback_on_step_end=progress_callback,
        ).images[0]
        
        # 动态命名：例如 result_image_1-3.png (代表第1组的第3张图)
        file_name = f"result_image_{prompt_id}-{i}.png"
        image.save(file_name)
        print(f"\n✅ 大功告成！图片已保存为 {file_name}")

print("\n🎉 全部任务执行完毕！")