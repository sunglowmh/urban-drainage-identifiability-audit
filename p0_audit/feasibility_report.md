# BlueHeartV2 数据可行性审计（小规模）

生成: 2026-11-06 15:08 ｜ 口径: 复用 p0_audit/gauge_index_summary.csv（994 文件逐文件实测）+ 空间复算

---

## 1. 各类型共同覆盖期

| 类型 | 站数 | 最早数据 | 最晚数据 |
|---|---|---|---|
| Borehole | 11 | 2022-07-13 | 2026-05-01 |
| Fluvial | 58 | 2021-11-09 | 2026-05-01 |
| Highway gully | 15 | 2025-06-30 | 2026-04-30 |
| Rain | 15 | 2024-02-19 | 2026-05-01 |
| Sewer | 398 | 2022-06-28 | 2026-05-01 |

- **五类共同覆盖窗口: 2025-06-30 → 2026-04-30**（受雨水口 2025-06-30 上线与 2026-04-30 终止约束，约 10 个月）
- 雨-河-污三类共同窗口: 2024-02-19 → 2026-05-01（约 26 个月）

五类窗口内活跃站数（该窗口内至少有一条记录）:

| 类型 | 活跃站数 / 总数 | 备注 |
|---|---|---|
| Borehole | 10/11 | |
| Fluvial | 55/58 | |
| Highway gully | 15/15 | |
| Rain | 14/15 | |
| Sewer | 330/398 | |

窗口内**全时段**在线（start ≤ 窗口起点 且 end ≥ 窗口终点）:

| 类型 | 全时段在线 / 总数 |
|---|---|
| Borehole | 0/11 |
| Fluvial | 39/58 |
| Highway gully | 0/15 |
| Rain | 14/15 |
| Sewer | 300/398 |

---

## 2. 采样间隔（逐站中位间隔的分布）

| 类型 | 中位间隔(s) | P10 | P90 | 站间一致性 |
|---|---|---|---|---|
| Borehole | 3600 | 900 | 3600 | 分散(雨开口径:事件触发) |
| Fluvial | 300 | 300 | 300 | 集中 |
| Highway gully | 943 | 913 | 944 | 集中 |
| Rain | 900 | 900 | 900 | 集中 |
| Sewer | 900 | 900 | 900 | 集中 |

注: 雨水口为事件触发式上报（15.2–15.7 min 名义间隔），联合分析前必须重采样/规则化；其余四类为规整等间隔。

---

## 3. 有效率（flag==1 占有标签数据点比例；缺失率单列）

| 类型 | flag1(Good)% | flag2(Suspect)% | flag3(Bad)% | 数值缺失% |
|---|---|---|---|---|
| Borehole | 75.6 | 23.4 | 1.0 | 1.2 |
| Fluvial | 94.3 | 4.8 | 0.9 | 24.9 |
| Highway gully | 100.0 | 0.0 | 0.0 | 22.7 |
| Rain | 100.0 | 0.0 | 0.0 | 0.0 |
| Sewer | 76.5 | 23.5 | 0.0 | 0.0 |

与前次 BlueHeart 工作区实测（84.46 / 15.13 / 0.40）一致；V2 数据集与 V1 同源同内容（行数逐类吻合）。
重点: 污水 23.5% 可疑几乎全为标签 B（雷达零点与管内基准不统一，负值≠干管），只可作相对变化分析；钻孔水温 E 标签为规则假阳性。

---

## 4. 站点坐标 ↔ 管网节点匹配（V2 复算）

| 类型 | 站数 | 最近节点距离中位(m) | <30m | <50m | <100m | 最近节点类型 top3 |
|---|---|---|---|---|---|---|
| Borehole | 11 | 58.0 | 36% | 45% | 82% | Manhole=8, Break=2, Outfall=1 |
| Fluvial | 58 | 6.4 | 83% | 86% | 91% | Break=43, Outfall=11, Storage=2 |
| Highway gully | 15 | 9.4 | 87% | 93% | 93% | Manhole=15 |
| Rain | 15 | 65.2 | 7% | 27% | 80% | Manhole=15 |
| Sewer | 398 | 3.8 | 93% | 96% | 99% | Manhole=391, Break=4, Storage=2 |

- 污水站绑定建议: 以 **30 m** 为阈值可绑定 372/398 站，其中最近节点为 Manhole 的 391 站；>30 m 的 26 站需人工核对。
- 索引经纬度仅 4 位小数（≈10 m 量级），30 m 阈值已接近坐标精度下限，属'可绑定但需抽查'。
- 雨量/钻孔站本来就不在管网上，匹配距离大是预期行为，仅作参照。


---

## 5. 雨水口 floodThreshold_mm 的语义（明确不当地表积水真值）

- 15 个雨水口阈值范围: 298–1437 mm，中位 720 mm。

| 雨水口 | floodThreshold_mm |
|---|---|
| ESH5B_Star Rd | 775 |
| ESH6A_Cornfield | 298 |
| ESH7C_Sheen Rd | 1136 |
| ESH8D_Terminus Rd | 685 |
| JBA11A_Aylesbury | 590 |
| JBA12A_Hereward Rd | 797 |
| JBA13C_Slindon | 903 |
| JBA15A_Montague Way | 1301 |
| JBA16A_Castle dr | 616 |
| JBA17A_Timberlaine Road | 800 |
| JBA18A_Swanley | 470 |
| JBA3A_Channel view Rd | 720 |
| JBA5A_Pevensey Rd | 713 |
| JBA6A_Hampden | 1437 |
| JBA9C_Bourne St | 356 |

**语义声明**: README 定义为 `water level above which flooding occurs`，即**该监测点（雨水口井腔）自身的漫溢阈值**，由 gully 视角（井腔水位达到阈值视为该点发生漫溢）给出，≠ 2D 地表淹没水深真值，且 15 个点远不构成面状积水观测。在论文中只能用作「点位级 gully 漫溢事件的二值标签/召回评估」，不能作为 LISFLOOD-FP 水深验证数据。

---

## 6. 可用于独立验证的地表积水记录核查

### 6.1 数据集内部

- `Study_area.shp` 字段: FID, geometry → 无积水/淹没类字段
- `Links.shp` 字段: id, us_node_id, ds_node_id, link_type, system_typ, shape, conduit_wi, gradient, capacity, us_invert, ds_invert, invert, diameter, discharge_, limiting_d, switch_on_, off_delay, discharge, base_level, width, overgate_d, opening, gate_depth, crest, height, angle, bar_spacin, length, geometry → 无积水/淹没类字段
- `Nodes.shp` 字段: id, DEPNOD, storage_ar, node_type, system_typ, ground_lev, flood_leve, geometry → 含 `flood_leve`

说明: `Nodes.flood_leve` 是**模型属性**（节点漫溢水位，前期实测与 ground_level 逐点相等），属管网资产参数而非观测记录；雨水口的 `floodThreshold_mm` 同理是监测点属性阈值（见 §5）。二者都不是'实际发生的地表积水'观测。

### 6.2 外部候选（按可得性排序）

| 来源 | 内容 | 用途 | 风险/备注 |
|---|---|---|---|
| Environment Agency **Historic Flood Map / Recorded Flood Outlines**（开放数据, data.gov.uk） | 英格兰历史记录洪水外包络面 | 2D 模型淹没范围的**粗粒度**独立验证 | 多为河洪记录，地表水积水轮廓稀疏；几何不精确 |
| EA **Hydrology Data Explorer** | 前期概要指出的 4 个 EA 河道站 + Pevensey 雨量站（不在包内） | 河道边界与降雨的独立对照 | 非积水数据，但可补齐边界真值 |
| Blue Heart 项目方（blueheart.org.uk / East Sussex CC）报告与事件记录 | 项目期内的实际内涝事件记载（如文献提到的 2015-08-23 39.4 mm 事件机制） | 案例事件的过程性验证 | 需向项目方申请/引用报告，非结构化 |
| 南方水司 Southern Water 内部 DG5/溢流记录 | 洪水上报记录 | 严格意义的积水验证 | **未公开**，需合作申请 |

**可行性结论**: 独立积水真值在包内为零，只能走「gully 阈值超限事件（点位级）+ EA 历史洪水轮廓（范围级）」两条弱验证路线，或争取 Southern Water/项目方数据。这决定了选题 1/2 的验证设计应以**管网/河道/地下水水位的时间过程匹配**为主战场，积水范围只作辅助校核。

---

## 7. 可行性判定

| 审计项 | 结论 | 依据 |
|---|---|---|
| 五类共同覆盖期 | **2025-06-30 → 2026-04-30（约 10 个月）**，3 类(雨河污)窗口约 26 个月 | §1 |
| 采样间隔 | 四类规整等间隔(5/15/60 min)，gully 事件触发需重采样 | §2 |
| 有效率 | Good 76–100%（污水最低 76.5% 且系统性 B 标签），Bad ≤1% | §3 |
| 站-网匹配 | 污水站 30 m 内 ~93% 可绑定 Manhole，坐标精度≈10 m | §4 |
| gully 阈值语义 | 仅点位漫溢阈值，不可作 2D 淹没真值 | §5 |
| 独立积水记录 | **包内为零**；外部仅 EA 历史洪水轮廓(粗)与项目方/SW 数据(需申请) | §6 |

**总判定**: 选题 1＋2（可辨识性＋多源同化）的数据基础成立，验证主轴应放在水位过程线（sewer/fluvial/gully/groundwater），事件库建议以 2025-07–2026-03 五类齐备窗口内、由 15 个雨量站圈定的暴雨事件为准。