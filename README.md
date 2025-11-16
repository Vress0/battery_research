# Na3PS4 固態電解質分子動力學模擬

使用 CHGNet 深度學習力場進行 Na3PS4 固態電解質的多溫度分子動力學模擬與輸運性質分析。

## 📋 專案概述

本專案系統性研究 Na3PS4 固態電解質在 400K、500K、600K 三個溫度下的離子傳輸行為，包含：
- **結構優化**：FIRE 演算法進行能量最小化
- **分子動力學模擬**：NVT 系綜，60 ps 軌跡
- **擴散分析**：平均平方位移 (MSD) 與擴散係數計算
- **結構分析**：徑向分佈函數 (RDF) 與配位環境
- **輸運性質**：離子導電率與 Arrhenius 活化能

## 🔬 系統資訊

**材料**：Na3PS4 (三硫磷酸鈉)
- 晶系：立方晶系
- 空間群：I-43m
- 超晶胞：2×2×2 (128 原子)
  - Na: 48 個
  - P: 16 個  
  - S: 64 個
- 晶胞體積：2714.08 Å³
- Na⁺ 濃度：0.0294 mol/cm³

## 📁 專案結構

```
battery_research/
├── README.md                          # 本文件
├── CHGNET/                            # CHGNet 模擬主資料夾
│   ├── 400K/                          # 400K 溫度模擬
│   │   ├── POSCAR/                    # 初始結構檔案
│   │   │   └── Na3PS4.poscar          # VASP 格式結構
│   │   ├── relax.py                   # 結構優化腳本
│   │   ├── md.py                      # 分子動力學模擬腳本
│   │   ├── convert.py                 # 軌跡格式轉換
│   │   ├── relaxed_fire_POSCAR        # 優化後的結構
│   │   ├── FIRE.traj / FIRE.xyz       # 優化軌跡
│   │   ├── md_out.traj / md_out.xyz   # MD 軌跡
│   │   ├── msd_400K.png               # MSD 圖表
│   │   └── rdf_400K.png               # RDF 圖表
│   ├── 500K/                          # 500K 溫度模擬
│   │   ├── analysis.py                # MSD/RDF 分析腳本
│   │   └── ...                        # (結構同 400K)
│   ├── 600K/                          # 600K 溫度模擬
│   │   └── ...                        # (結構同 400K)
│   ├── conductivity_analysis.py       # 離子導電率分析腳本
│   ├── arrhenius_plot.png            # Arrhenius 圖
│   └── conductivity_comparison.png    # 導電率比較圖
```

## 📝 使用說明

### 環境需求

```bash
pip install chgnet pymatgen ase numpy matplotlib scipy
```

### 快速開始

#### 1. 結構優化
```bash
cd CHGNET/400K
python relax.py
```

#### 2. 分子動力學模擬
```bash
python md.py
```

#### 3. 軌跡格式轉換
```bash
python convert.py  # 將 .traj 轉為 .xyz
```

#### 4. MSD/RDF 分析
```bash
cd CHGNET/500K
python analysis.py
```
輸出：`msd_500K.png`、`rdf_500K.png`

#### 5. 導電率分析
```bash
cd CHGNET
python conductivity_analysis.py
```
輸出：`arrhenius_plot.png`、`conductivity_comparison.png`

## 📊 主要研究成果

### 1. 擴散係數 (從 MSD 計算)

| 溫度 | Na 擴散係數 (m²/s) | Na 擴散係數 (cm²/s) | 相對 400K |
|------|-------------------|-------------------|----------|
| 400K | 2.08×10⁻¹⁰        | 2.08×10⁻⁶         | 1.0×     |
| 500K | 2.88×10⁻⁹         | 2.88×10⁻⁵         | 13.8×    |
| 600K | 5.64×10⁻⁹         | 5.64×10⁻⁵         | 27.1×    |

**重點**：P 和 S 的擴散係數接近零，形成穩定的 PS₄³⁻ 骨架。

### 2. 離子導電率 (Nernst-Einstein 關係)

| 溫度 | 離子導電率 (S/cm) | 離子導電率 (mS/cm) |
|------|-----------------|-------------------|
| 400K | 0.171           | 171               |
| 500K | 1.89            | 1,890             |
| 600K | 3.09            | 3,090             |

### 3. Arrhenius 活化能

- **活化能 (Ea)**: 0.350 eV (350 meV)
- **預指數因子 (σ₀)**: 2.09×10⁶ S·K/cm
- **物理意義**：Na⁺ 在晶格中跳躍所需克服的能障

### 4. 結構特徵 (RDF 分析)

主要原子對的第一配位殼層：

| 配對  | 第一峰位置 (Å) | 溫度趨勢          |
|-------|---------------|------------------|
| Na-S  | 2.78-2.83     | 峰高隨溫度微降    |
| Na-P  | ~3.53         | 配位數 ~6，穩定  |
| Na-Na | ~3.53         | 峰高降低，序度降 |

## 🔍 分析方法

### MSD (Mean Square Displacement)

計算原子位移的平方平均值:

$$\text{MSD}(t) = \langle |\mathbf{r}(t) - \mathbf{r}(0)|^2 \rangle$$

從 MSD 的線性區域斜率計算擴散係數:

$$\text{MSD} = 6Dt$$

### RDF (Radial Distribution Function)

計算原子對的距離分佈,用於分析結構特性:

$$g(r) = \frac{V}{N^2} \left\langle \sum_{i}\sum_{j\neq i} \delta(r - r_{ij}) \right\rangle$$

### 離子導電率

使用 Nernst-Einstein 關係從擴散係數計算導電率:

$$\sigma = \frac{nq^2D}{Vk_BT}$$

其中:
- n: 離子數量
- q: 電荷 (Na⁺ = 1e)
- D: 擴散係數
- V: 體積
- k_B: Boltzmann 常數
- T: 溫度

### Arrhenius 分析

導電率的溫度依賴性:

$$\sigma T = \sigma_0 \exp\left(-\frac{E_a}{k_BT}\right)$$

通過 ln(σT) vs 1000/T 的線性擬合得到活化能 Ea。

## 💡 關鍵結論

### 擴散機制
- **Na⁺ 為主要載子**：擴散係數比 P/S 高 4-5 個數量級
- **溫度依賴性強**：400K → 600K，擴散係數增加 27 倍
- **Arrhenius 行為**：ln(D) vs 1/T 呈線性，符合熱活化擴散

### 離子導電率
- **500K 性能優異**：達 1.89 S/cm，為鈉固態電解質高值
- **溫度效應顯著**：400K → 600K，導電率增加 18 倍
- **實用潛力**：活化能 0.35 eV 適中，實際應用可行

### 結構穩定性
- **PS₄³⁻ 骨架穩定**：P 和 S 無明顯擴散，維持框架完整性
- **Na-S 配位**：第一殼層 2.78-2.83 Å，典型配位距離
- **熱擾動效應**：溫度升高使 RDF 峰高降低，局部序度下降

## 🛠️ 計算方法

### CHGNet 力場
- **模型類型**：圖神經網路通用原子勢
- **模型版本**：CHGNet v0.3+
- **訓練數據**：Materials Project DFT 資料庫
- **精度**：接近 DFT，速度快 3-4 個數量級
- **計算設備**：GPU 加速 (CUDA)

### 分子動力學參數
- **系綜**：NVT (Nosé-Hoover 恆溫器)
- **溫度**：400K, 500K, 600K
- **時間步長**：1 fs
- **總時長**：60 ps (60,000 步)
- **輸出頻率**：每 10 步記錄一次
- **輸出文件**：md_out.traj (51 MB), md_out.xyz (129 MB)

### 結構優化
- **演算法**：FIRE (Fast Inertial Relaxation Engine)
- **收斂標準**：fmax < 0.05 eV/Å
- **最大迭代**：1000 步 (400K) / 200 步 (500K, 600K)

### MSD 分析方法
- **計算公式**：$\text{MSD}(t) = \langle |\mathbf{r}(t) - \mathbf{r}(0)|^2 \rangle$
- **擴散係數**：$D = \frac{1}{6} \lim_{t \to \infty} \frac{d}{dt}\text{MSD}(t)$
- **邊界處理**：週期性邊界條件，最小鏡像法

### RDF 分析方法
- **計算公式**：$g(r) = \frac{V}{N^2} \langle \sum_{i \neq j} \delta(r - |\mathbf{r}_{ij}|) \rangle$
- **徑向範圍**：0–6 Å
- **殼層厚度**：0.1 Å
- **配位數計算**：積分至第一極小值

### 導電率計算
- **Nernst-Einstein 方程**：$\sigma = \frac{n q^2 D}{V k_B T}$
- **Haven ratio**：假設 $H_R = 1$
- **Arrhenius 活化能**：$\sigma T = \sigma_0 \exp(-E_a / k_B T)$

### 週期性邊界條件
所有計算採用三維週期性邊界條件 (PBC)，正確處理超晶胞邊界的原子運動。

## 📚 參考資料

- CHGNet: [https://github.com/CederGroupHub/chgnet](https://github.com/CederGroupHub/chgnet)
- ASE (Atomic Simulation Environment): [https://wiki.fysik.dtu.dk/ase/](https://wiki.fysik.dtu.dk/ase/)
- Pymatgen: [https://pymatgen.org/](https://pymatgen.org/)

## 📝 注意事項

1. 模擬使用 CHGNet 預訓練模型,精度接近 DFT 但速度快得多
2. 結果適用於理解趨勢和機制,精確數值可能需要更高階的計算驗證
3. MD 模擬時間 (60 ps) 對於導電率計算是合理的,但更長時間可能提高統計精度

## 👤 作者

Ryan (Vress0)

## 📄 授權

本專案僅供學術研究使用。

---

**最後更新**: 2025年11月13日
