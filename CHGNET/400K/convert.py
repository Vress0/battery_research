import os
from ase.io import read, write

# 找到目前資料夾裡所有 .traj 檔案
for file in os.listdir("."):
    if file.endswith(".traj"):
        xyz_file = file.replace(".traj", ".xyz")
        print(f"正在轉換 {file} → {xyz_file}")
        frames = read(file, ":")   # 讀取全部幀
        write(xyz_file, frames)    # 輸出成 xyz 格式
        print(f"完成：{xyz_file}")
