# Battery Research - Na3PS4 固態電解質模擬

使用 CHGNet 深度學習力場模型進行 Na3PS4 (三硫磷酸鈉) 固態電解質的分子動力學模擬與離子導電率分析。

## 📋 專案簡介

本專案研究 Na3PS4 固態電解質在不同溫度 (400K, 500K, 600K) 下的離子傳輸性質,包括:
- 結構優化與弛豫
- 分子動力學 (MD) 模擬
- 平均平方位移 (MSD) 分析
- 徑向分佈函數 (RDF) 分析
- 離子導電率計算
- Arrhenius 活化能分析

## 🔬 研究材料

**Na3PS4** - 一種很有前景的鈉離子固態電解質材料
- 化學式: Na₃PS₄
- 結構: 立方晶系
- 超晶胞: 2×2×2 擴展 (128 原子)
- 組成: 48 Na, 16 P, 64 S

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

## 🚀 使用方法

### 1. 環境需求

```bash
# Python 套件
pip install chgnet
pip install pymatgen
pip install ase
pip install numpy
pip install matplotlib
pip install scipy
```

### 2. 結構優化

在各溫度資料夾中執行:

```bash
cd CHGNET/400K
python relax.py
```

使用 FIRE 演算法優化結構:
- 收斂標準: fmax = 0.05 eV/Å
- 最大步數: 1000 (400K) / 200 (其他)

### 3. 分子動力學模擬

```bash
python md.py
```

模擬參數:
- 系綜: NVT (正則系綜)
- 溫度: 400K / 500K / 600K
- 時間步長: 1 fs
- 總步數: 60,000 (60 ps)
- 記錄間隔: 每 10 步

### 4. 軌跡格式轉換

將 `.traj` 格式轉換為 `.xyz` 格式以便視覺化:

```bash
python convert.py
```

### 5. MSD 與 RDF 分析

計算平均平方位移和徑向分佈函數:

```bash
cd CHGNET/500K
python analysis.py
```

輸出:
- `msd_XXX.png`: MSD 圖表
- `rdf_XXX.png`: RDF 圖表
- 擴散係數計算

### 6. 離子導電率分析

分析所有溫度並計算導電率:

```bash
cd CHGNET
python conductivity_analysis.py
```

輸出:
- `arrhenius_plot.png`: Arrhenius 圖
- `conductivity_comparison.png`: 導電率比較
- 活化能計算

## 📊 主要結果

### 離子導電率 (Ionic Conductivity)

| 溫度 | 擴散係數 (cm²/s) | 離子導電率 (mS/cm) | 導電率 (S/cm) |
|------|-----------------|-------------------|--------------|
| 400K | 2.08×10⁻⁶       | 171               | 0.171        |
| 500K | 2.88×10⁻⁵       | 1,890             | 1.89         |
| 600K | 5.64×10⁻⁵       | 3,090             | 3.09         |

### 活化能 (Activation Energy)

- **Ea = 0.350 eV** (350 meV)
- 預指數因子: σ₀ = 2.09×10⁶ S·K/cm

### 物理參數

- Na⁺ 離子數量: 48 (在超晶胞中)
- Na⁺ 濃度: 2.94×10⁻² mol/cm³
- 超晶胞體積: 2714.08 Å³

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

## 💡 關鍵發現

1. **優異的離子導電率**: 
   - 在 500K 達到 1.89 S/cm
   - 在 600K 達到 3.09 S/cm
   - 顯示 Na3PS4 是優秀的固態電解質材料

2. **合理的活化能**:
   - Ea = 0.350 eV 在典型鈉離子導體範圍內
   - 表明 Na⁺ 離子可以相對容易地在晶格中移動

3. **強溫度依賴性**:
   - 從 400K 到 600K,導電率增加約 18 倍
   - 符合 Arrhenius 熱活化傳輸機制

4. **Na⁺ 為主要載子**:
   - Na 的擴散係數遠大於 P 和 S
   - PS₄³⁻ 骨架保持相對穩定

## 🛠️ 技術細節

### CHGNet 模型

- 基於圖神經網路的力場模型
- 在 GPU 上運行以加速計算
- 可預測能量、力、應力和磁矩

### 優化演算法

- **FIRE** (Fast Inertial Relaxation Engine)
- 適合結構優化的快速方法

### 週期性邊界條件

所有計算都考慮週期性邊界條件 (PBC),正確處理超晶胞邊界的原子運動。

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
