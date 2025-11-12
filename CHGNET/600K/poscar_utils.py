# poscar_utils.py
import os
import glob
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

def smart_load_poscar(poscar_path):
    """智能載入 POSCAR，可處理檔案或目錄"""
    
    if os.path.isfile(poscar_path):
        # 明確指定為 POSCAR 格式
        return Poscar.from_file(poscar_path).structure
    
    elif os.path.isdir(poscar_path):
        patterns = [
            os.path.join(poscar_path, "*.poscar"),
            os.path.join(poscar_path, "*.vasp"),
            os.path.join(poscar_path, "POSCAR*"),
            os.path.join(poscar_path, "CONTCAR*"),
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                print(f"找到檔案: {files[0]}")
                # 使用 Poscar 明確讀取
                return Poscar.from_file(files[0]).structure
        
        all_files = [f for f in os.listdir(poscar_path) 
                    if os.path.isfile(os.path.join(poscar_path, f))]
        if all_files:
            file_path = os.path.join(poscar_path, all_files[0])
            print(f"使用第一個檔案: {all_files[0]}")
            # 使用 Poscar 明確讀取
            return Poscar.from_file(file_path).structure
    
    raise FileNotFoundError(f"無法在 {poscar_path} 中找到有效的 POSCAR 檔案")
