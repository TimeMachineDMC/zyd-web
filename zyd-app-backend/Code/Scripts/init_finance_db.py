import os
import pandas as pd
import sqlite3

DATA_DIR = r"..\..\Data"
DB_PATH = r"..\..\Model\finance_data.db"

def init_database():
    print(f"开始创建本地 SQLite 财务数据库: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(".xlsx") and not file.startswith("~$"):
                file_path = os.path.join(root, file)
                
                table_name = os.path.splitext(file)[0]
                
                print(f"正在抽取并导入表: {table_name} ...")
                try:
                    df = pd.read_excel(file_path, nrows=1000)
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                    print(f"  成功导入 {len(df)} 行数据至表 `{table_name}`")
                except Exception as e:
                    print(f"  导入表 {table_name} 失败: {e}")
                    
    conn.close()
    print("\n🎉 全部抽样数据入库完成！")

if __name__ == "__main__":
    init_database()