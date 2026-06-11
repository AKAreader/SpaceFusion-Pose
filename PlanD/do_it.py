import os
import shutil
import pandas as pd
import glob
import re

# ================= 配置区域 =================
source_root_dir = r"E:\Data"
output_root_dir = r"D:\BaiduNetdiskDownload\All"

# 这里的名称保持原样即可，代码会自动尝试匹配空格或下划线
target_folders = [
    "Aqua 60度",
    "Clementine_10度",
    "CubeSat_150度",
    "Landsat180度",
    "oco2_120度"
]

FRAMES_TO_TAKE = 1000  # 保持和你需求一致


# ===========================================

def main():
    print("=== 开始严格对齐检查 (增强版) ===")

    # 1. 扫描所有 Excel
    all_excels = glob.glob(os.path.join(source_root_dir, "*.xlsx")) + \
                 glob.glob(os.path.join(source_root_dir, "*.xls"))

    # 2. 准备输出目录
    out_rgb = os.path.join(output_root_dir, "All_RGB")
    out_ir = os.path.join(output_root_dir, "All_IR")
    if not os.path.exists(out_rgb): os.makedirs(out_rgb)
    if not os.path.exists(out_ir): os.makedirs(out_ir)

    global_counter = 1
    all_dfs = []

    for folder_name in target_folders:
        print(f"\n正在处理任务: [{folder_name}]")

        # --- 步骤 A: 修正文件夹路径 (关键修复) ---
        # 尝试直接路径
        full_path = os.path.join(source_root_dir, folder_name)

        # 如果找不到，尝试把空格换成下划线 (Aqua 60度 -> Aqua_60度)
        if not os.path.exists(full_path):
            alt_name = folder_name.replace(" ", "_")
            alt_path = os.path.join(source_root_dir, alt_name)
            if os.path.exists(alt_path):
                print(f"  -> [自动修正] 列表写的是 '{folder_name}'，但实际文件夹是 '{alt_name}'。已修正路径。")
                full_path = alt_path
                folder_name = alt_name  # 更新名称以便后续匹配
            else:
                print(f"  -> [严重错误] 找不到文件夹: {full_path}")
                print(f"     也不是: {alt_path}")
                print("     请检查 E:\\Data 下到底叫什么名字。")
                continue  # 彻底跳过

        # --- 步骤 B: 找 Excel ---
        key_match = re.split(r'[\d\s_]+', folder_name)[0].lower()
        if "landsat" in folder_name.lower(): key_match = "landsat"

        found_excel = None
        # 优先完全匹配
        for excel_path in all_excels:
            if folder_name in os.path.basename(excel_path):
                found_excel = excel_path;
                break
        # 其次模糊匹配
        if not found_excel:
            for excel_path in all_excels:
                if key_match in os.path.basename(excel_path).lower():
                    found_excel = excel_path;
                    break

        if not found_excel:
            print(f"  -> [跳过] 找不到对应的 Excel 文件，无法对齐。")
            continue

        # --- 步骤 C: 读取 Excel ---
        try:
            df = pd.read_excel(found_excel, engine='openpyxl')
        except Exception as e:
            print(f"  -> [错误] Excel 读取失败: {e}")
            continue

        actual_rows = len(df)

        # --- 步骤 D: 检查图片文件夹 ---
        src_rgb = os.path.join(full_path, "RGB")
        src_ir = os.path.join(full_path, "IR")

        if not os.path.exists(src_rgb):
            print(f"  -> [错误] 找不到 RGB 文件夹！路径: {src_rgb}")
            continue  # 如果没图片文件夹，就跳过

        # 获取图片列表
        images = sorted(os.listdir(src_rgb))
        img_count = len(images)

        # --- 步骤 E: 计算最终截取数量 ---
        # 取三者最小值：Excel行数、图片数、设定上限
        valid_count = min(actual_rows, img_count, FRAMES_TO_TAKE)

        print(f"  -> 状态确认: Excel行数={actual_rows}, 图片数={img_count}")
        print(f"  -> 最终执行截取: {valid_count} 组")

        # 截取 Excel
        df_subset = df.iloc[:valid_count].copy()
        df_subset['Original_Folder'] = folder_name
        all_dfs.append(df_subset)

        # 截取并复制图片
        images_to_process = images[:valid_count]
        processed_count = 0

        for img_name in images_to_process:
            s_rgb = os.path.join(src_rgb, img_name)
            s_ir = os.path.join(src_ir, img_name)

            if os.path.exists(s_ir):
                ext = os.path.splitext(img_name)[1]
                new_name = f"{global_counter:04d}{ext}"

                shutil.copy2(s_rgb, os.path.join(out_rgb, new_name))
                shutil.copy2(s_ir, os.path.join(out_ir, new_name))

                global_counter += 1
                processed_count += 1
            else:
                # 只有找不到IR的时候才报警告
                # print(f"警告: 缺少 IR 图 {img_name}")
                pass

        if processed_count < valid_count:
            print(f"  -> [修正] 因缺少IR图片，实际只处理了 {processed_count} 张。正在裁剪 Excel...")
            all_dfs[-1] = all_dfs[-1].iloc[:processed_count]

        print(f"  -> [{folder_name}] 处理完毕。")
        print("-" * 30)

    # --- 保存 ---
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.insert(0, 'Unified_ID', [f"{i:04d}" for i in range(1, len(final_df) + 1)])
        out_excel = os.path.join(output_root_dir, "All_Data.xlsx")
        final_df.to_excel(out_excel, index=False)
        print(f"\n全部完成！总数据量: {len(final_df)} 行")
        print(f"结果保存在: {output_root_dir}")
    else:
        print("\n[失败] 没有生成任何数据。")


if __name__ == "__main__":
    main()