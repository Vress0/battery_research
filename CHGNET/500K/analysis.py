import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from scipy.spatial.distance import pdist, squareform
import os

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def calculate_msd(trajectory, species=None, dt=1.0):
    """
    計算平均平方位移 (MSD)
    
    Parameters:
    -----------
    trajectory : list of Atoms
        MD 軌跡
    species : str or None
        指定元素符號 (例如 'Na')，None 表示計算所有原子
    dt : float
        時間步長 (fs)
    
    Returns:
    --------
    times : array
        時間陣列 (ps)
    msd : array
        MSD 值 (Å²)
    """
    n_frames = len(trajectory)
    
    # 選擇要分析的原子
    if species:
        indices = [i for i, atom in enumerate(trajectory[0]) 
                  if atom.symbol == species]
        print(f"分析 {species} 原子，共 {len(indices)} 個")
    else:
        indices = range(len(trajectory[0]))
        print(f"分析所有原子，共 {len(indices)} 個")
    
    # 獲取初始位置
    positions = np.array([trajectory[i].get_positions()[indices] 
                         for i in range(n_frames)])
    
    # 處理週期性邊界條件
    cell = trajectory[0].get_cell()
    for i in range(1, n_frames):
        delta = positions[i] - positions[i-1]
        # 處理跨越邊界的情況
        delta = delta - np.round(delta / cell.diagonal()) * cell.diagonal()
        positions[i] = positions[i-1] + delta
    
    # 計算 MSD
    msd = np.zeros(n_frames)
    for lag in range(n_frames):
        if lag == 0:
            msd[lag] = 0
        else:
            # 計算所有可能的時間原點
            n_origins = n_frames - lag
            displacements = positions[lag:] - positions[:-lag] if lag < n_frames else positions[-1:] - positions[:1]
            squared_displacements = np.sum(displacements**2, axis=2)
            msd[lag] = np.mean(squared_displacements)
    
    times = np.arange(n_frames) * dt / 1000  # 轉換為 ps
    
    return times, msd


def calculate_diffusion_coefficient(times, msd, fit_range=(0.3, 0.7)):
    """
    從 MSD 計算擴散係數
    
    Parameters:
    -----------
    times : array
        時間陣列 (ps)
    msd : array
        MSD 值 (Å²)
    fit_range : tuple
        用於線性擬合的時間範圍 (相對於總時間的比例)
    
    Returns:
    --------
    D : float
        擴散係數 (cm²/s)
    """
    # 選擇線性區域進行擬合
    start_idx = int(len(times) * fit_range[0])
    end_idx = int(len(times) * fit_range[1])
    
    # 線性擬合: MSD = 6Dt
    coeffs = np.polyfit(times[start_idx:end_idx], msd[start_idx:end_idx], 1)
    slope = coeffs[0]  # Å²/ps
    
    # 轉換單位: Å²/ps -> cm²/s
    # 1 Å² = 1e-16 cm², 1 ps = 1e-12 s
    D = slope / 6 * 1e-16 / 1e-12 * 1e12  # cm²/s
    
    return D, coeffs


def calculate_rdf(trajectory, rmax=10.0, nbins=200, species_pair=None):
    """
    計算徑向分佈函數 (RDF)
    
    Parameters:
    -----------
    trajectory : list of Atoms
        MD 軌跡
    rmax : float
        最大距離 (Å)
    nbins : int
        分格數量
    species_pair : tuple or None
        指定元素對 (例如 ('Na', 'S'))，None 表示計算所有原子對
    
    Returns:
    --------
    r : array
        距離陣列 (Å)
    g_r : array
        RDF 值
    """
    dr = rmax / nbins
    r = np.linspace(0, rmax, nbins)
    g_r = np.zeros(nbins)
    
    n_frames = len(trajectory)
    
    for frame in trajectory:
        positions = frame.get_positions()
        cell = frame.get_cell()
        volume = frame.get_volume()
        
        # 選擇要分析的原子
        if species_pair:
            indices_i = [i for i, atom in enumerate(frame) 
                        if atom.symbol == species_pair[0]]
            indices_j = [i for i, atom in enumerate(frame) 
                        if atom.symbol == species_pair[1]]
            
            for i in indices_i:
                for j in indices_j:
                    if i != j or species_pair[0] != species_pair[1]:
                        # 考慮週期性邊界條件
                        delta = positions[j] - positions[i]
                        delta = delta - np.round(delta / cell.diagonal()) * cell.diagonal()
                        distance = np.linalg.norm(delta)
                        
                        if distance < rmax:
                            bin_idx = int(distance / dr)
                            if bin_idx < nbins:
                                g_r[bin_idx] += 1
            
            n_i = len(indices_i)
            n_j = len(indices_j) if species_pair[0] != species_pair[1] else len(indices_j) - 1
            normalization = n_i * n_j * n_frames
        else:
            # 計算所有原子對
            n_atoms = len(positions)
            for i in range(n_atoms):
                for j in range(i+1, n_atoms):
                    delta = positions[j] - positions[i]
                    delta = delta - np.round(delta / cell.diagonal()) * cell.diagonal()
                    distance = np.linalg.norm(delta)
                    
                    if distance < rmax:
                        bin_idx = int(distance / dr)
                        if bin_idx < nbins:
                            g_r[bin_idx] += 2  # 計入 i-j 和 j-i
            
            normalization = n_atoms * (n_atoms - 1) * n_frames
        
        # 正規化
        for i in range(nbins):
            r_inner = i * dr
            r_outer = (i + 1) * dr
            shell_volume = 4/3 * np.pi * (r_outer**3 - r_inner**3)
            number_density = normalization / volume
            g_r[i] /= (shell_volume * number_density)
    
    return r, g_r


def plot_msd(times, msd_dict, output_file='msd_plot.png', temperature=500):
    """
    繪製 MSD 圖
    
    Parameters:
    -----------
    times : array
        時間陣列 (ps)
    msd_dict : dict
        {species: msd} 字典
    output_file : str
        輸出檔案名稱
    temperature : float
        溫度 (K)
    """
    plt.figure(figsize=(10, 6))
    
    for species, msd in msd_dict.items():
        plt.plot(times, msd, label=species, linewidth=2)
        
        # 計算擴散係數
        if len(times) > 10:
            D, coeffs = calculate_diffusion_coefficient(times, msd)
            print(f"{species} 擴散係數: {D:.2e} cm²/s")
            
            # 繪製擬合線
            fit_range = (int(len(times)*0.3), int(len(times)*0.7))
            fit_line = coeffs[0] * times[fit_range[0]:fit_range[1]] + coeffs[1]
            plt.plot(times[fit_range[0]:fit_range[1]], fit_line, 
                    '--', alpha=0.5, label=f'{species} 擬合 (D={D:.2e} cm²/s)')
    
    plt.xlabel('時間 (ps)', fontsize=12)
    plt.ylabel(r'MSD ($\AA^2$)', fontsize=12)
    plt.title(f'平均平方位移 (MSD) - {temperature}K', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"MSD 圖已儲存至: {output_file}")
    plt.close()


def plot_rdf(r, rdf_dict, output_file='rdf_plot.png', temperature=500):
    """
    繪製 RDF 圖
    
    Parameters:
    -----------
    r : array
        距離陣列 (Å)
    rdf_dict : dict
        {pair_name: g_r} 字典
    output_file : str
        輸出檔案名稱
    temperature : float
        溫度 (K)
    """
    plt.figure(figsize=(10, 6))
    
    for pair_name, g_r in rdf_dict.items():
        plt.plot(r, g_r, label=pair_name, linewidth=2)
    
    plt.xlabel(r'距離 r ($\AA$)', fontsize=12)
    plt.ylabel('g(r)', fontsize=12)
    plt.title(f'徑向分佈函數 (RDF) - {temperature}K', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, r[-1])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"RDF 圖已儲存至: {output_file}")
    plt.close()


def main():
    # 讀取軌跡檔案
    print("正在讀取軌跡檔案...")
    traj_file = "md_out.traj"
    
    if not os.path.exists(traj_file):
        print(f"錯誤: 找不到 {traj_file}")
        return
    
    trajectory = read(traj_file, ":")
    print(f"成功讀取 {len(trajectory)} 幀")
    
    # 取得溫度資訊 (從資料夾名稱)
    current_dir = os.path.basename(os.getcwd())
    temperature = int(current_dir.replace('K', '')) if 'K' in current_dir else 500
    
    # 取得元素種類
    symbols = set(trajectory[0].get_chemical_symbols())
    print(f"包含的元素: {symbols}")
    
    # ==================== 計算 MSD ====================
    print("\n" + "="*50)
    print("計算 MSD (平均平方位移)")
    print("="*50)
    
    msd_dict = {}
    dt = 1.0  # 時間步長 (fs)
    
    # 計算每種元素的 MSD
    for species in sorted(symbols):
        print(f"\n計算 {species} 的 MSD...")
        times, msd = calculate_msd(trajectory, species=species, dt=dt)
        msd_dict[species] = msd
    
    # 計算所有原子的 MSD
    print(f"\n計算所有原子的 MSD...")
    times, msd_all = calculate_msd(trajectory, species=None, dt=dt)
    msd_dict['All'] = msd_all
    
    # 繪製 MSD
    plot_msd(times, msd_dict, f'msd_{temperature}K.png', temperature)
    
    # ==================== 計算 RDF ====================
    print("\n" + "="*50)
    print("計算 RDF (徑向分佈函數)")
    print("="*50)
    
    rdf_dict = {}
    
    # 計算特定原子對的 RDF (例如 Na-S, Na-P, S-S 等)
    if 'Na' in symbols and 'S' in symbols:
        print("\n計算 Na-S RDF...")
        r, g_r = calculate_rdf(trajectory, rmax=10.0, nbins=200, 
                              species_pair=('Na', 'S'))
        rdf_dict['Na-S'] = g_r
    
    if 'Na' in symbols and 'P' in symbols:
        print("\n計算 Na-P RDF...")
        r, g_r = calculate_rdf(trajectory, rmax=10.0, nbins=200, 
                              species_pair=('Na', 'P'))
        rdf_dict['Na-P'] = g_r
    
    if 'Na' in symbols:
        print("\n計算 Na-Na RDF...")
        r, g_r = calculate_rdf(trajectory, rmax=10.0, nbins=200, 
                              species_pair=('Na', 'Na'))
        rdf_dict['Na-Na'] = g_r
    
    # 繪製 RDF
    if rdf_dict:
        plot_rdf(r, rdf_dict, f'rdf_{temperature}K.png', temperature)
    
    print("\n" + "="*50)
    print("分析完成!")
    print("="*50)


if __name__ == "__main__":
    main()
