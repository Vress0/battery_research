import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import os

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 物理常數
k_B = 1.380649e-23  # Boltzmann constant (J/K)
e = 1.602176634e-19  # Elementary charge (C)
N_A = 6.02214076e23  # Avogadro's number


def calculate_msd_for_conductivity(trajectory, species='Na', dt=1.0):
    """
    計算指定元素的 MSD (用於導電率計算)
    
    Parameters:
    -----------
    trajectory : list of Atoms
        MD 軌跡
    species : str
        元素符號
    dt : float
        時間步長 (fs)
    
    Returns:
    --------
    times : array
        時間陣列 (ps)
    msd : array
        MSD 值 (Å²)
    n_ions : int
        離子數量
    """
    n_frames = len(trajectory)
    
    # 選擇要分析的原子
    indices = [i for i, atom in enumerate(trajectory[0]) 
              if atom.symbol == species]
    n_ions = len(indices)
    
    print(f"分析 {species} 原子，共 {n_ions} 個")
    
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
            n_origins = n_frames - lag
            displacements = positions[lag:] - positions[:-lag] if lag < n_frames else positions[-1:] - positions[:1]
            squared_displacements = np.sum(displacements**2, axis=2)
            msd[lag] = np.mean(squared_displacements)
    
    times = np.arange(n_frames) * dt / 1000  # 轉換為 ps
    
    return times, msd, n_ions


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
        擴散係數 (m²/s)
    slope : float
        擬合斜率
    intercept : float
        擬合截距
    """
    # 選擇線性區域進行擬合
    start_idx = int(len(times) * fit_range[0])
    end_idx = int(len(times) * fit_range[1])
    
    # 線性擬合: MSD = 6Dt
    coeffs = np.polyfit(times[start_idx:end_idx], msd[start_idx:end_idx], 1)
    slope = coeffs[0]  # Å²/ps
    intercept = coeffs[1]
    
    # 轉換單位: Å²/ps -> m²/s
    # 1 Å² = 1e-20 m², 1 ps = 1e-12 s
    D = slope / 6 * 1e-20 / 1e-12  # m²/s
    
    return D, slope, intercept


def calculate_ionic_conductivity(D, n_ions, volume, temperature, charge=1):
    """
    使用 Nernst-Einstein 關係計算離子導電率
    
    σ = (n * q² * D) / (V * k_B * T)
    
    Parameters:
    -----------
    D : float
        擴散係數 (m²/s)
    n_ions : int
        離子數量
    volume : float
        超晶胞體積 (Å³)
    temperature : float
        溫度 (K)
    charge : int
        離子電荷 (通常 Na+ = 1)
    
    Returns:
    --------
    sigma : float
        離子導電率 (S/cm)
    """
    # 轉換體積: Å³ -> m³
    volume_m3 = volume * 1e-30
    
    # 計算導電率 (S/m)
    sigma_SI = (n_ions * (charge * e)**2 * D) / (volume_m3 * k_B * temperature)
    
    # 轉換為 S/cm (常用單位)
    sigma_S_cm = sigma_SI / 100
    
    return sigma_S_cm


def calculate_concentration(n_ions, volume):
    """
    計算離子濃度
    
    Parameters:
    -----------
    n_ions : int
        離子數量
    volume : float
        體積 (Å³)
    
    Returns:
    --------
    concentration : float
        濃度 (mol/cm³)
    """
    # 轉換體積: Å³ -> cm³
    volume_cm3 = volume * 1e-24
    
    # 計算濃度
    concentration = (n_ions / N_A) / volume_cm3
    
    return concentration


def analyze_temperature(temp_folder, species='Na', dt=1.0):
    """
    分析單一溫度資料夾
    
    Parameters:
    -----------
    temp_folder : str
        溫度資料夾路徑
    species : str
        要分析的離子種類
    dt : float
        時間步長 (fs)
    
    Returns:
    --------
    result : dict
        分析結果
    """
    # 讀取軌跡
    traj_file = os.path.join(temp_folder, "md_out.traj")
    
    if not os.path.exists(traj_file):
        print(f"警告: 找不到 {traj_file}")
        return None
    
    print(f"\n分析 {temp_folder}")
    print("="*60)
    
    trajectory = read(traj_file, ":")
    n_frames = len(trajectory)
    print(f"讀取 {n_frames} 幀")
    
    # 取得溫度
    folder_name = os.path.basename(temp_folder)
    temperature = int(folder_name.replace('K', ''))
    print(f"溫度: {temperature} K")
    
    # 取得體積
    volume = trajectory[0].get_volume()  # Å³
    print(f"超晶胞體積: {volume:.2f} Å³")
    
    # 計算 MSD
    times, msd, n_ions = calculate_msd_for_conductivity(trajectory, species=species, dt=dt)
    
    # 計算擴散係數
    D, slope, intercept = calculate_diffusion_coefficient(times, msd)
    D_cm2_s = D * 1e4  # m²/s -> cm²/s
    
    print(f"\n{species} 離子數量: {n_ions}")
    print(f"{species} 擴散係數: {D:.2e} m²/s ({D_cm2_s:.2e} cm²/s)")
    
    # 計算離子導電率
    sigma = calculate_ionic_conductivity(D, n_ions, volume, temperature, charge=1)
    print(f"{species}⁺ 離子導電率: {sigma:.2e} S/cm ({sigma*1000:.2e} mS/cm)")
    
    # 計算濃度
    concentration = calculate_concentration(n_ions, volume)
    print(f"{species}⁺ 濃度: {concentration:.2e} mol/cm³")
    
    # 計算遷移率 (mobility): μ = D * e / (k_B * T)
    mobility = D * e / (k_B * temperature)
    print(f"{species}⁺ 遷移率: {mobility:.2e} m²/(V·s)")
    
    result = {
        'temperature': temperature,
        'n_ions': n_ions,
        'volume': volume,
        'D_m2_s': D,
        'D_cm2_s': D_cm2_s,
        'sigma_S_cm': sigma,
        'sigma_mS_cm': sigma * 1000,
        'concentration': concentration,
        'mobility': mobility,
        'times': times,
        'msd': msd,
        'slope': slope,
        'intercept': intercept
    }
    
    return result


def plot_arrhenius(results_list, output_file='arrhenius_plot.png'):
    """
    繪製 Arrhenius plot (ln(σT) vs 1000/T)
    
    用於計算活化能: σT = σ₀ exp(-Ea/kT)
    ln(σT) = ln(σ₀) - Ea/(kT)
    """
    if len(results_list) < 2:
        print("需要至少兩個溫度點才能繪製 Arrhenius plot")
        return
    
    temperatures = np.array([r['temperature'] for r in results_list])
    sigmas = np.array([r['sigma_S_cm'] for r in results_list])
    
    # 計算 σT 和 1000/T
    sigma_T = sigmas * temperatures
    inv_T_1000 = 1000 / temperatures
    
    # 線性擬合
    coeffs = np.polyfit(inv_T_1000, np.log(sigma_T), 1)
    slope = coeffs[0]  # -Ea/k
    intercept = coeffs[1]  # ln(σ₀)
    
    # 計算活化能
    # slope = -Ea/k_B, 需要轉換為 eV
    Ea_J = -slope * k_B * 1000  # J
    Ea_eV = Ea_J / e  # eV
    
    print("\n" + "="*60)
    print("Arrhenius 分析結果")
    print("="*60)
    print(f"活化能 (Ea): {Ea_eV:.3f} eV ({Ea_J*1000:.2f} meV)")
    print(f"預指數因子 (σ₀): {np.exp(intercept):.2e} S·K/cm")
    
    # 繪圖
    plt.figure(figsize=(10, 6))
    
    # 實驗數據
    plt.scatter(inv_T_1000, np.log(sigma_T), s=100, c='red', 
                marker='o', label='模擬數據', zorder=5)
    
    # 擬合線
    fit_line = slope * inv_T_1000 + intercept
    plt.plot(inv_T_1000, fit_line, 'b--', linewidth=2, 
             label=f'擬合 (Ea = {Ea_eV:.3f} eV)')
    
    # 標註溫度
    for i, T in enumerate(temperatures):
        plt.annotate(f'{T}K', 
                    (inv_T_1000[i], np.log(sigma_T[i])),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=10)
    
    plt.xlabel('1000/T (K$^{-1}$)', fontsize=12)
    plt.ylabel(r'ln($\sigma$T) [ln(S$\cdot$K/cm)]', fontsize=12)
    plt.title(f'Arrhenius Plot - Na+ 離子導電率\nEa = {Ea_eV:.3f} eV', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nArrhenius plot 已儲存至: {output_file}")
    plt.close()
    
    return Ea_eV


def plot_conductivity_comparison(results_list, output_file='conductivity_comparison.png'):
    """
    繪製不同溫度下的導電率比較
    """
    temperatures = [r['temperature'] for r in results_list]
    sigmas_mS_cm = [r['sigma_mS_cm'] for r in results_list]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左圖: 導電率 vs 溫度 (線性尺度)
    ax1.plot(temperatures, sigmas_mS_cm, 'bo-', linewidth=2, markersize=10)
    for T, sigma in zip(temperatures, sigmas_mS_cm):
        ax1.annotate(f'{sigma:.2f}', 
                    (T, sigma),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=9)
    ax1.set_xlabel('溫度 (K)', fontsize=12)
    ax1.set_ylabel('離子導電率 (mS/cm)', fontsize=12)
    ax1.set_title('Na+ 離子導電率 vs 溫度', fontsize=13)
    ax1.grid(True, alpha=0.3)
    
    # 右圖: 導電率 vs 溫度 (對數尺度)
    ax2.semilogy(temperatures, sigmas_mS_cm, 'ro-', linewidth=2, markersize=10)
    for T, sigma in zip(temperatures, sigmas_mS_cm):
        ax2.annotate(f'{sigma:.2e}', 
                    (T, sigma),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=9)
    ax2.set_xlabel('溫度 (K)', fontsize=12)
    ax2.set_ylabel('離子導電率 (mS/cm) [log scale]', fontsize=12)
    ax2.set_title('Na+ 離子導電率 vs 溫度 (對數尺度)', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"導電率比較圖已儲存至: {output_file}")
    plt.close()


def main():
    # 設定工作目錄
    base_dir = r"C:\Users\Ryan\Desktop\GitHub\battery_research\CHGNET"
    
    # 溫度資料夾列表
    temp_folders = ['400K', '500K', '600K']
    
    print("="*60)
    print("離子導電率分析")
    print("="*60)
    
    # 分析每個溫度
    results_list = []
    for temp_folder in temp_folders:
        folder_path = os.path.join(base_dir, temp_folder)
        result = analyze_temperature(folder_path, species='Na', dt=1.0)
        if result:
            results_list.append(result)
    
    if not results_list:
        print("錯誤: 沒有可分析的數據")
        return
    
    # 輸出摘要表格
    print("\n" + "="*60)
    print("導電率摘要")
    print("="*60)
    print(f"{'溫度 (K)':<12} {'擴散係數 (cm²/s)':<20} {'導電率 (mS/cm)':<20}")
    print("-"*60)
    for r in results_list:
        print(f"{r['temperature']:<12} {r['D_cm2_s']:<20.2e} {r['sigma_mS_cm']:<20.2e}")
    
    # 繪製 Arrhenius plot
    if len(results_list) >= 2:
        arrhenius_file = os.path.join(base_dir, 'arrhenius_plot.png')
        Ea_eV = plot_arrhenius(results_list, arrhenius_file)
    
    # 繪製導電率比較圖
    conductivity_file = os.path.join(base_dir, 'conductivity_comparison.png')
    plot_conductivity_comparison(results_list, conductivity_file)
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)


if __name__ == "__main__":
    main()
