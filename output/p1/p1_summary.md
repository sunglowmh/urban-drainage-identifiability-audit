# P1 阶段汇总报告（2026-11-08）

## 交付物一览

| 任务 | 交付物 | 状态 |
|---|---|---|
| P1.1 基线骨架 | `p1_build_swmm.py` → `output/p1/blueheart_baseline.inp`（全网络 v0）+ `build_report.md` | ✅ 冒烟通过（水量误差 0.0%） |
| P1.2 潮信号 | `p1_tidal.py` → `output/p1/tidal_report.md` + `fluvial_tidal_detection.csv` | ✅ 负发现：规则潮信号可忽略 |
| P1.3 试点选择 | `p1_pilot.py` → `output/p1/pilot_selection.md` + 节点/传感器清单 | ✅ |
| P1.4 预注册 | `research_route/preregistration.md` v1.0 | ✅ |
| P1.5 产流面+试点模型 | `p1_pilot_model.py` → `output/p1/blueheart_pilot.inp` + `pilot_subcatchments.csv` | ✅ 事件模拟跑通 |
| P1.6 30 站筛选 | `output/p1/pilot_30_stations.csv` + `pilot_sim_vs_obs_30stations.csv` | ✅ |
| P1.7 泵站评估 | 本文档 §泵站 | ✅ 评估完成，暂不接入 |

## P1.5 产流面与试点模型

- **产流基元**：infiltration 2D zones（30,259 个落在试点内，`area` 单位公顷）；按就近 junction 聚合为 **2,018 个 subcatchments（合计 4.50 km²）**；
- **%Imperv** = Σ(area×runoff_coe)/Σarea（Fixed 型区）；Horton 参数（initial≈50 mm / limiting≈1 mm/h / decay≈2 /h）面积加权进入 [INFILTRATION]；
- **坡度**：elevation 2D zones（38.5 万个单元，bbox 裁剪读取）按节点分组 p90−p10 / 2√A，夹取 [0.5%, 10%]；
- **伪出流**：死端节点（有入流无出流）82 个设 FREE outfall——注意这会**高估**排水能力（泵站湿井未模拟）；
- **雨量**：15 站面平均，RG1 VOLUME 15min 序列；
- 事件模拟（2025-10-22，38.4 mm，36 h 含缓冲）：DYNWAVE，**水量路由误差 0.0%**。

### 试点模型结构（v1）
2,274 conduits / 2,238 junctions / 15 真实 outfall + 82 伪出流 / 16 虚拟汇流节点 / 2,018 subcatchments。未含：泵 13 台、堰/孔口、DWF 旱季污水流。

## P1.6 30 站筛选与首轮模拟-实测对比

- 筛选规则（预注册精神）：试点 102 活跃站按**质量有效率（flag1/value_cells）主序 × 记录量次序**取 30；入选站有效率 0.991–1.000；
- 首轮未校准对比（2025-10-22 事件，15 min 对齐，29/30 站有效）：
  - Pearson r：中位 **0.44**，均值 0.34；
  - Spearman ρ：中位 **0.55**；
- **判读**：未校准骨架（统一 n=0.013、无 DWF、无泵、伪出流自由出流）能复现约半数站点的水位过程主要动态，起点合格；低相关站点的候选原因（结构性）：DWF 缺失（foul 系统）、泵站截流、支管连接缺失、传感器基准面差异（观测 mm 深度 vs 模拟节点水深，未做 datum 对齐）。**下一步优先级**：① DWF 近似（按节点汇水面积比例分配旱季基流）② datum 对齐规则 ③ 低相关站逐站归因。

## P1.7 泵站评估

- 全网 95 台 Pump：`switch_on_level` / `base_level` 覆盖 100%，但**泵送能力（discharge）0%**、off_delay 1%——**数据不足以构建泵特性曲线**；
- 试点子网内 13 台，当前未接入；
- 处置：① 阶段一推断维持无泵（结构误差已在 §判读 声明）；② P2 如需，泵能力以"满管 Manning 流量"为先验、开关水位用实测 switch_on_level，做情景敏感性；③ 泵站运行记录可向 East Sussex/Southern Water 申请，列为数据增补项。

## 转换器踩坑记录（复用必读）

1. `FLOW_UNITS`（非 FLOW_UNIT）；XSECT 须写 `CIRCULAR`；
2. 节点 ID 含空格 → clean_id 净化并同步 us/ds 映射；
3. Storage 节点 FUNCTIONAL 参数列歧义 → 降级 Junction；
4. Outfall 多入流 → 虚拟汇流节点 `OUTJ_*` + 单连 conduit `DUMMY_*`；
5. **过流型 Outfall**（`*_CI` 涵洞入口等，作为 conduit 上游）→ 转 Junction；
6. **死端判定**：`in ds_nodes & ~in us_nodes`（只有入流无出流），勿写成 `~in ds_nodes`（那是源头）；
7. infiltration `area` 单位 = **公顷**；雨量序列用 RAINGAGES `VOLUME 0:15`；
8. 节点 invert/ground 缺失 → 修正为 (0, 1.5) 保连通；MaxDepth 夹取 [0.1, 50] m；
9. 长度用 EPSG:27700 几何长度（属性 length 仅 3.4% 覆盖）；elevation shp（503 万面）用 bbox 裁剪读取。

## P1 → P2 门槛状态

- [x] 基线模型连通性正确（冒烟 0.0% 误差）
- [x] 产流-汇流-出流链路事件模拟跑通
- [x] 30 个高质量观测站锁定（推断观测输入）
- [x] 预注册文档 v1.0（判级阈值固定）
- [ ] 代理模型构建与后验稳定性验证（阶段一推断跑通后）
- [ ] 合成孪生校准（依赖代理）
- [ ] DWF 近似与 datum 对齐（影响模拟-实测一致性，建议代理构建前完成）
