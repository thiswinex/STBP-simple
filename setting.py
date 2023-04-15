STEPS = 2                           # 时间步数
DT = 5                              # 时间步长，仅在时序（DVS等）数据集下有意义
SIMWIN = DT * STEPS # 仿真时间窗口
ALPHA = 0.5                         # 梯度近似项
VTH = 0.2                           # 阈值电压 V_threshold
TAU = 0.25                          # 漏电常数 tau