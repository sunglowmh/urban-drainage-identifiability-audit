# Blue Heart 数据集 P0 审计报告

生成时间: 2026-11-06 14:53  |  数据根目录: E:\work\workBuddy\BlueHeartV2\dataset

---

## 1. 测点索引 (index_telemetry_data.csv)

- 总测点数: **497**
- 按类型: Sewer=398, Fluvial=58, Rain=15, Highway gully=15, Borehole=11
- 按属主: Southern Water=398, Blue Heart=99
- floodThreshold_mm 非空: 15 (仅 Highway gully 期望有)
- bedLevel/bankLevel 非空: 39 / 39 (仅 Fluvial 期望有)

| 类型 | 参数(单位) 组合 |
|---|---|
| Borehole | Air Temperature [m]; Depth To Water [mAOD]; Water Level [°C] |
| Fluvial | Air Temperature [m]; Water Depth [mAOD]; Water Level [°C] |
| Highway gully | Signal-to-noise ratio [dB]; Water Level [mm] |
| Rain | Rainfall Depth [mm] |
| Sewer | Sewer Level [mm] |

- 参数/单位列表长度不一致的测点: 0
- dataPath 或 flagsPath 缺失的测点: 0

---

## 2. 矢量数据 (catchment / network shapefiles)

### Study_area (研究区多边形)
- CRS: EPSG:32610  |  要素数: 1
- 属性列: FID
- 总面积(投影后): 56.280 km^2
- Bounds: [np.float64(4249191.8), np.float64(12678669.8), np.float64(4261983.9), np.float64(12689628.2)]

### 2D Zones elevation
- CRS: EPSG:4326  |  要素数: 5035742
- 属性列: element_no, elevation2
- Bounds: [np.float64(0.2), np.float64(50.7), np.float64(0.4), np.float64(50.8)]

### 2D Zones infiltration
- CRS: EPSG:4326  |  要素数: 188123
- 属性列: polygon_id, area, infiltrati, runoff_vol, initial_in, limiting_i, decay_fact, recovery_f, runoff_coe
- Bounds: [np.float64(0.2), np.float64(50.7), np.float64(0.4), np.float64(50.9)]

 infiltration surfaces metadata:

- surface_id=2D zone (outside of infiltration zones); runoff_volume_type=Horton; initial_infiltration=50.0; limiting_infiltration=1.0; decay_factor=2.0; recovery_factor=1.0; runoff_coefficient=nan
- surface_id=Buildings outside subcatchments; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.85
- surface_id=DEFAULT!!; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=1.0
- surface_id=EBO - 40% SPRHOST; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.1168
- surface_id=EBO - 70% SPRHOST; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.2336
- surface_id=EBO Roads 10%; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.1
- surface_id=EBO building 25%; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.25
- surface_id=EBO/HAIL - 10% SPRHOST; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.029
- surface_id=EBO/HAIL - 20% SPRHOST; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.058
- surface_id=EBO/HAIL - 30% SPRHOST; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.088
- surface_id=EBO/HAIL - 50% SPRHOST; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.146
- surface_id=EBO/HAIL - 60% SPRHOST; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.175
- surface_id=Gardens in subcatchments; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.07
- surface_id=HAIL Buildings 15%; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.15
- surface_id=HAIL Roads 25%; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.25
- surface_id=Inland Water; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.0
- surface_id=JBA SW SC Buildings; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.15
- surface_id=JBA SW SC Roads; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.25
- surface_id=Roads outside subcatchments; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.75
- surface_id=Roads/Roofs in subcatchments; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.25
- surface_id=Rural WRAP 1; runoff_volume_type=Horton; initial_infiltration=200.0; limiting_infiltration=1.0; decay_factor=2.0; recovery_factor=1.0; runoff_coefficient=nan
- surface_id=Rural WRAP 3; runoff_volume_type=Horton; initial_infiltration=44.0; limiting_infiltration=1.0; decay_factor=2.0; recovery_factor=0.5; runoff_coefficient=nan
- surface_id=Rural WRAP 4; runoff_volume_type=Horton; initial_infiltration=44.0; limiting_infiltration=1.0; decay_factor=2.0; recovery_factor=1.0; runoff_coefficient=nan
- surface_id=Urban; runoff_volume_type=Fixed; initial_infiltration=nan; limiting_infiltration=nan; decay_factor=nan; recovery_factor=nan; runoff_coefficient=0.7

- infiltration zones metadata 行数: 188123, surface_id 取值: ['Buildings outside subcatchments', 'EBO - 40% SPRHOST', 'EBO - 70% SPRHOST', 'EBO Roads 10%', 'EBO building 25%', 'EBO/HAIL - 10% SPRHOST', 'EBO/HAIL - 20% SPRHOST', 'EBO/HAIL - 30% SPRHOST', 'EBO/HAIL - 50% SPRHOST', 'EBO/HAIL - 60% SPRHOST', 'HAIL Buildings 15%', 'HAIL Roads 25%', 'Inland Water', 'JBA SW SC Buildings', 'JBA SW SC Roads', 'Roads outside subcatchments', 'Rural WRAP 1', 'Rural WRAP 3', 'Rural WRAP 4']

### Links (管网/河道 links)
- CRS: EPSG:4326  |  要素数: 15267
- 属性列: id, us_node_id, ds_node_id, link_type, system_typ, shape, conduit_wi, gradient, capacity, us_invert, ds_invert, invert, diameter, discharge_, limiting_d, switch_on_, off_delay, discharge, base_level, width, overgate_d, opening, gate_depth, crest, height, angle, bar_spacin, length
- Bounds: [np.float64(0.2), np.float64(50.8), np.float64(0.4), np.float64(50.9)]

### Nodes (管网节点)
- CRS: EPSG:4326  |  要素数: 14843
- 属性列: id, DEPNOD, storage_ar, node_type, system_typ, ground_lev, flood_leve
- Bounds: [np.float64(0.2), np.float64(50.8), np.float64(0.4), np.float64(50.9)]

---

**研究区面积复核**: Study_area.shp 总面积 = **56.28 km^2**（与你的高程网格估计 ~56 km^2 一致；论文口径 78 km^2 为更大集水区，二者不矛盾，论文中需明确区分 study-area polygon vs. catchment）。

**区外基础设施**: Nodes 共 14843, 位于 study_area 多边形内 13311 个, 区外 1532 个 (10.3%)。
- Links 共 15267, 质心在区外 1527 条 (10.0%)。

---

## 3. 时序遥测数据 (994 CSV 审计)

逐文件统计：记录数、时间覆盖、中位采样间隔、质量标记分布（flag 1/2/3 与字母标签 A–E）、缺失值比例。

| 类型 | 文件数 | 总记录 | 覆盖起点 | 覆盖终点 | 中位采样间隔(s) | 缺失值% | flag1(Good)% | flag2(Suspect)% | flag3(Bad)% | 无时区偏移文件数 |
|---|---|---|---|---|---|---|---|---|---|---|
| Borehole | 11 | 215,626 | 2022-07-13 | 2026-05-01 | 3600.0 | 1.2% | 74.7% | 23.1% | 1.0% | 0 |
| Fluvial | 58 | 12,947,377 | 2021-11-09 | 2026-05-01 | 300.0 | 24.9% | 70.8% | 3.6% | 0.7% | 0 |
| Sewer | 398 | 36,555,178 | 2022-06-28 | 2026-05-01 | 900.0 | 0.0% | 76.5% | 23.5% | 0.0% | 0 |
| Highway gully | 15 | 412,489 | 2025-06-30 | 2026-04-30 | 943.0 | 22.7% | 77.3% | 0.0% | 0.0% | 15 |
| Rain | 15 | 930,470 | 2024-02-19 | 2026-05-01 | 900.0 | 0.0% | 100.0% | 0.0% | 0.0% | 0 |

质量字母标签统计（逐值）:

- **Borehole**: B=5,515, D=4,648, E=129,221
- **Fluvial**: A=156,173, B=446,604, D=119,253, E=946,966
- **Sewer**: B=8,034,588, D=72,999, E=580,112
- **Highway gully**: B=109, C=192, D=6
- **Rain**: 

**时区核查**: Highway gully 原始数据 timestamp 不带 UTC 偏移的文件数 = 15/15；其余类型全部带偏移（README 口径一致）。质量标签文件一律带 +00:00 —— 对齐时必须先对 gully 原始时间戳做显式 localize/convert，不可直接混合解析。

---

## 4. 初步数据风险提示

- gully 采样为设备事件触发（非整点等间隔），同化前需重采样/规则化。
- Sewer Level 单位为 mm（非 mAOD），Fluvial/Borehole 为 mAOD —— 建模时注意量纲与基准面。
- index 中 Sewer 测点多属 Southern Water（WPS 泵站命名），可能含泵站启停伪影（rate-of-change flag D 应能捕捉）。
- infiltration 元数据为 Horton 参数（initial/limiting/decay/recovery），即官方 1D-2D 模型的入渗参数集——参数可辨识性实验（选题1）的先验范围可直接以此为起点。

---
