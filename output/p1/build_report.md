# P1.1 SWMM 基线骨架构建报告 (v0)

- 输入: 15267 links / 14843 nodes
- Conduit(CIRC) 保留: **13219** (全部 link_type=Conduit 13443, CIRC 13222)
- Junction: 13328, Outfall: 589, Storage(未入 v0): 49
- 剔除 conduit: {'conduit_bad_size': 0, 'conduit_bad_len': 3, 'conduit_bad_invert': 0, 'conduit_outfall_as_us': 0}
- 剔除非 Conduit link: 1824 (Weir 217, Pump 95, Orifice 52, River reach 475 等; v0 不含水力构筑物)
- 长度来源: EPSG:27700 几何长度(属性 length 仅 3% 覆盖)
- 粗糙度: 统一初值 0.013 (后续为可辨识性参数, 分区化在 P1 后续)
- 输出: E:/work/workBuddy/BlueHeartV2\output\p1\blueheart_baseline.inp
