import os
import pandas as pd

DATA_DIR = r"..\..\Data\数据"
OUTPUT_FILE = "headers.txt"

print("开始极速扫描所有 Excel 表头，请稍候...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            # 只处理 xlsx，且过滤掉系统临时文件
            if file.endswith(".xlsx") and not file.startswith("~$"):
                file_path = os.path.join(root, file)
                try:
                    # nrows=0 是核心：只读表头，不读数据
                    df = pd.read_excel(file_path, nrows=0)
                    headers = list(df.columns)
                    
                    # 写入到 txt 中
                    f.write(f"【文件路径】: {os.path.relpath(file_path, DATA_DIR)}\n")
                    f.write(f"【真实表头】: {headers}\n")
                    f.write("-" * 50 + "\n\n")
                    
                    print(f"成功提取: {file}")
                except Exception as e:
                    print(f"读取失败: {file}, 报错: {e}")
                    f.write(f"【文件路径】: {os.path.relpath(file_path, DATA_DIR)}\n")
                    f.write(f"【读取失败】: {e}\n")
                    f.write("-" * 50 + "\n\n")

print(f"\n全部表头提取完成！结果已保存至同目录下的 {OUTPUT_FILE}")