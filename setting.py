DEFAULT_STEPS = 2                           # 时间步数
DEFAULT_DT = 5                              # 时间步长，仅在时序（DVS等）数据集下有意义
DEFAULT_SIMWIN = DEFAULT_DT * DEFAULT_STEPS # 仿真时间窗口
DEFAULT_ALPHA = 0.5                         # 梯度近似项
DEFAULT_VTH = 0.2                           # 阈值电压 V_threshold
DEFAULT_TAU = 0.25                          # 漏电常数 tau