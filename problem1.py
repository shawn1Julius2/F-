# -*- coding: utf-8 -*-
"""
问题一（航班直接分配给机组人员）的可运行 Python 源代码
======================================================

本代码严格实现“修正版”数学模型（第 1 问），核心特性如下：
1) 目标字典序：① 最大化覆盖的航班数量；②④（本问只含④）在覆盖最优面上最小化乘机次数；③⑤⑥不在本问；⑦在第二层之后最小化替补使用次数。
2) 乘机-执行绑定：未起飞的航班不能配置任何机组（包含乘机），即 y_{p,f} ≤ z_f 且 ∑_p y_{p,f} ≤ MaxDH·z_f。
3) 单路径-基地闭环：每名机组的全期路径是从基地出发并最终回到基地的一条“单位流”单路径。
4) 资格字段联动：Captain/FirstOfficer/Deadhead 字段与变量域一致绑定；仅 Captain 且具备 FirstOfficer 替补资格者可以执行 x^{Fsub}。
5) 可连接弧按“同机场 + 最小连接时间（MinCT=40min）”生成，源/汇弧只允许基地起止。
6) 代码默认读取 A 套数据（可切换 B 套），输出结果文件：UncoveredFlights.csv 与 CrewRosters.csv，并自动生成覆盖率、乘机次数分布、替补使用分布等可视化 PNG 图片，保存并显示；中文与负号显示已修正。

依赖：
- pandas, numpy, matplotlib, pulp（内置 CBC 优先；若无 CBC 则使用 pulp 自带求解器）
- 代码采用相对路径读取当前目录下的 CSV 文件：
  机组：  机组排班Data A-Crew.csv / 机组排班Data B-Crew.csv
  航班：  机组排班Data A-Flight.csv / 机组排班Data B-Flight.csv

注意：
- B 套规模很大（约 1.4 万航班），若内存/时间受限，建议先用 A 套验证流程。
- 为避免弧规模爆炸，构图采用“按机场分组 + 二分起点 + 限制后继窗口”的稀疏策略，可通过 MAX_SUCCESSORS 调参。
"""

import os
import sys
import math
import time
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 中文与负号显示（按题主要求仅需两句）
mpl.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
mpl.rcParams['axes.unicode_minus'] = False

try:
    import pulp
except ImportError as e:
    raise RuntimeError("运行本代码需要安装 PuLP：pip install pulp") from e

# ========== 配置区 ==========
DATASET = "A"  # 可选 "A" 或 "B"
MIN_CT_MINUTES = 40            # 最小连接时间（分钟）
MAX_DEADHEAD_PER_FLIGHT = 5    # 每航班最大乘机人数
MAX_SUCCESSORS = 20            # 每个航班在同机场的最多后继航班窗口（控制弧数量；A 套 10~30 较稳健）
RANDOM_SEED = 42               # 随机数种子（如需随机打散后继窗口）
SHOW_PLOTS = True              # 显示图形
SAVE_PLOTS = True              # 保存图形
RESULT_DIR = "."               # 结果输出目录（当前目录）

# 文件名（按 DATASET 切换）
CREW_FILE = f"机组排班Data {DATASET}-Crew.csv"
FLIGHT_FILE = f"机组排班Data {DATASET}-Flight.csv"

# ========== 工具函数：鲁棒字段识别 ==========
def pick_col(df, candidates, required=True, default=None):
    """
    在 DataFrame 列名中鲁棒地选择匹配列；按候选列表逐个尝试（大小写/空白不敏感）。
    candidates: 可能的列名候选（中文/英文）
    """
    norm = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm:
            return norm[key]
    # 再做一次“包含匹配”（防止前后空格/符号）
    for c in df.columns:
        c0 = str(c).strip().lower()
        for cand in candidates:
            if str(cand).strip().lower() in c0:
                return c
    if required:
        raise KeyError(f"缺少必要字段，候选：{candidates}")
    return default

def parse_bool_series(s):
    """
    将多种形式（Y/N, 1/0, True/False, 是/否）转为 0/1 整型 Series。
    """
    mapping = {
        'y':1,'yes':1,'true':1,'1':1,'是':1,'可':1,'可以':1,'允许':1,
        'n':0,'no':0,'false':0,'0':0,'否':0,'不可':0,'不可以':0,'禁止':0
    }
    def _map(v):
        if pd.isna(v): return 0
        if isinstance(v,(int,np.integer)): return int(v!=0)
        s = str(v).strip().lower()
        return mapping.get(s, 1 if s in ('c','f','cf') else 0)  # 兜底：'C','F','CF' 视为正样
    return s.map(_map).astype(int)

def parse_comp_to_req(comp_str):
    """
    将类似 'C1F1' 的最低配置字符串解析为 (req_C, req_F)。
    若已提供数值列，则不使用本函数。
    """
    if pd.isna(comp_str):
        return 0,0
    s = str(comp_str).upper().replace(" ","")
    c = 0; f = 0
    # 简单解析：找到 C 后的数字、F 后的数字
    import re
    mC = re.search(r'C(\d+)', s)
    mF = re.search(r'F(\d+)', s)
    if mC: c = int(mC.group(1))
    if mF: f = int(mF.group(1))
    return c,f

def to_minutes(ts):
    """
    将任意可解析时间字段转为“统一分钟戳”（整数，单位分钟）。
    支持：datetime, 'YYYY-MM-DD HH:MM' 等。
    """
    if pd.isna(ts):
        return None
    if isinstance(ts, (int, np.integer, float, np.floating)):
        return int(ts)
    if isinstance(ts, pd.Timestamp):
        return int(ts.value // 10**9 // 60)
    try:
        t = pd.to_datetime(ts)
        return int(t.value // 10**9 // 60)
    except Exception:
        # 若本就为分钟戳字符串
        try:
            return int(str(ts))
        except Exception:
            return None

# ========== 读取数据 ==========
if not os.path.exists(CREW_FILE) or not os.path.exists(FLIGHT_FILE):
    raise FileNotFoundError(f"未找到数据文件：{CREW_FILE} 或 {FLIGHT_FILE}（请将 CSV 放到当前目录）")

crew_raw = pd.read_csv(CREW_FILE)
flt_raw  = pd.read_csv(FLIGHT_FILE)

print("原始机组数据列：", list(crew_raw.columns))
print("原始航班数据列：", list(flt_raw.columns))

# 机组字段映射
col_crew_id  = pick_col(crew_raw, ["CrewID","机组编号","人员编号","ID"])
col_base     = pick_col(crew_raw, ["Base","基地","驻地"])
col_cap      = pick_col(crew_raw, ["Captain","正机长","C资格","是否正机长"])
col_fo       = pick_col(crew_raw, ["FirstOfficer","副机长","F资格","是否副机长"])
col_dh       = pick_col(crew_raw, ["Deadhead","乘机允许","是否允许乘机","可乘机"])
# Duty/Pairing 成本字段可能不存在于第 1 问，但为统一格式做兼容（若不存在则置零或缺省）
col_duty     = pick_col(crew_raw, ["DutyCostPerHr","Duty_Cost","执勤成本/小时","执勤成本"], required=False, default=None)

crew = pd.DataFrame({
    "CrewID": crew_raw[col_crew_id].astype(str),
    "Base": crew_raw[col_base].astype(str),
    "Captain": parse_bool_series(crew_raw[col_cap]),
    "FirstOfficer": parse_bool_series(crew_raw[col_fo]),
    "Deadhead": parse_bool_series(crew_raw[col_dh]),
})
if col_duty and col_duty in crew_raw.columns:
    crew["DutyCostPerHr"] = pd.to_numeric(crew_raw[col_duty], errors="coerce").fillna(0.0)
else:
    crew["DutyCostPerHr"] = 0.0

# 航班字段映射
col_flid    = pick_col(flt_raw, ["FlightID","航班编号","FltID","ID"])
col_depstn  = pick_col(flt_raw, ["DepStn","出发机场","起飞机场","Origin"])
col_arrstn  = pick_col(flt_raw, ["ArrStn","到达机场","降落机场","Destination"])
col_deptime = pick_col(flt_raw, ["DepTime","计划起飞时间","起飞时间","STD"])
col_arrtime = pick_col(flt_raw, ["ArrTime","计划到达时间","到达时间","STA"])

# 可能有 (req_C, req_F) 或一个 Comp 字段
req_C_col = None; req_F_col = None; comp_col = None
try:
    req_C_col = pick_col(flt_raw, ["req_C","最低配置C","C需求","NeedC"])
    req_F_col = pick_col(flt_raw, ["req_F","最低配置F","F需求","NeedF"])
except:
    comp_col = pick_col(flt_raw, ["Comp","最低配置","配置需求","C?F?"])

flights = pd.DataFrame({
    "FlightID": flt_raw[col_flid].astype(str),
    "DepStn": flt_raw[col_depstn].astype(str),
    "ArrStn": flt_raw[col_arrstn].astype(str),
    "DepTime_min": flt_raw[col_deptime].apply(to_minutes),
    "ArrTime_min": flt_raw[col_arrtime].apply(to_minutes),
})

if req_C_col and req_F_col:
    flights["req_C"] = pd.to_numeric(flt_raw[req_C_col], errors="coerce").fillna(0).astype(int)
    flights["req_F"] = pd.to_numeric(flt_raw[req_F_col], errors="coerce").fillna(0).astype(int)
else:
    reqs = flt_raw[comp_col].apply(parse_comp_to_req)
    flights["req_C"] = reqs.map(lambda x: x[0]).astype(int)
    flights["req_F"] = reqs.map(lambda x: x[1]).astype(int)

# 基础校验
if flights["DepTime_min"].isna().any() or flights["ArrTime_min"].isna().any():
    raise ValueError("存在无法解析的起降时间，请确保时间字段为可解析格式（或分钟戳）。")

# 排序与索引
flights = flights.sort_values(["DepStn","DepTime_min","ArrTime_min","FlightID"]).reset_index(drop=True)
flights["idx"] = np.arange(len(flights))  # 模型内部索引
crew = crew.reset_index(drop=True)
crew["pidx"] = np.arange(len(crew))

print("\n标准化后的机组数据预览：")
print(crew.head(10).to_string(index=False))
print("\n标准化后的航班数据预览：")
print(flights.head(10).to_string(index=False))

# ========== 构建可连接弧（按机场分组 + 二分查找 + 限制后继窗口） ==========
from bisect import bisect_left

np.random.seed(RANDOM_SEED)

# 按机场分组
grouped = {stn: df for stn, df in flights.groupby("DepStn", as_index=False)}
# 为每个机场建立“按 DepTime 排序”的辅助结构，以搜索后继
airport_dep_index = {}
for stn, df in grouped.items():
    arr = df["DepTime_min"].values.tolist()
    airport_dep_index[stn] = (arr, df)

# 对每个航班 f，在 f.ArrStn 的出发序列里找 DepTime >= ArrTime + MIN_CT_MINUTES 的最早位置
A_links = []  # (f_idx, g_idx)
for _, f in flights.iterrows():
    arr_stn = f["ArrStn"]
    arr_t   = f["ArrTime_min"]
    if arr_stn not in airport_dep_index:
        continue
    dep_list, dep_df = airport_dep_index[arr_stn]
    t0 = arr_t + MIN_CT_MINUTES
    pos = bisect_left(dep_list, t0)
    # 限制窗口大小，避免弧爆炸；窗口内按起飞时间升序选前 MAX_SUCCESSORS 个
    stop = min(len(dep_list), pos + MAX_SUCCESSORS)
    if pos < stop:
        # 左开右闭 [pos, stop)
        g_block = dep_df.iloc[pos:stop]
        for _, g in g_block.iterrows():
            A_links.append((int(f["idx"]), int(g["idx"])))

A_links = list(set(A_links))  # 去重
print(f"\n可连接弧数量（不含源/汇弧）：{len(A_links):,}")

# 为每名机组添加源/汇弧 + 闲置零长弧
# S_p -> f: 仅 DepStn == Base
# f -> T_p: 仅 ArrStn == Base
# 以及 S_p -> T_p 零长弧
# 我们将这些“弧类型”在变量命名中编码，以便区分
# - 普通弧：('F', f_idx, g_idx)
# - 源弧：('S', pidx, f_idx)
# - 汇弧：('T', pidx, f_idx)   我们用 (f -> T) 的方向记成 ('T',p,f)
# - 闲置弧：('ST', pidx)      即 S_p -> T_p

# 预计算每名机组的源/汇可连航班 idx 集合
crew_source_targets = {}
crew_sink_sources   = {}
for _, p in crew.iterrows():
    base = p["Base"]
    pidx = int(p["pidx"])
    # 从基地出发的航班
    f_src = flights.loc[flights["DepStn"]==base, "idx"].astype(int).tolist()
    # 到基地到达的航班
    f_sink = flights.loc[flights["ArrStn"]==base, "idx"].astype(int).tolist()
    crew_source_targets[pidx] = f_src
    crew_sink_sources[pidx]   = f_sink

# ========== 建立 MILP 模型 ==========
prob = pulp.LpProblem("CrewAssignment_Q1", pulp.LpMaximize)

# 变量：覆盖 z_f
z = {int(f["idx"]): pulp.LpVariable(f"z_f{int(f['idx'])}", cat=pulp.LpBinary)
     for _, f in flights.iterrows()}

# 变量：角色/替补/乘机 x/y  与 节点使用 u
xC, xF, xFsub, y, u = {}, {}, {}, {}, {}
for _, p in crew.iterrows():
    pidx = int(p["pidx"])
    for _, f in flights.iterrows():
        fidx = int(f["idx"])
        xC[(pidx,fidx)]    = pulp.LpVariable(f"xC_p{pidx}_f{fidx}",    cat=pulp.LpBinary)
        xF[(pidx,fidx)]    = pulp.LpVariable(f"xF_p{pidx}_f{fidx}",    cat=pulp.LpBinary)
        xFsub[(pidx,fidx)] = pulp.LpVariable(f"xFsub_p{pidx}_f{fidx}", cat=pulp.LpBinary)
        y[(pidx,fidx)]     = pulp.LpVariable(f"y_p{pidx}_f{fidx}",     cat=pulp.LpBinary)
        u[(pidx,fidx)]     = pulp.LpVariable(f"u_p{pidx}_f{fidx}",     cat=pulp.LpBinary)

# 变量：弧流 w（普通弧 + 源弧 + 汇弧 + 闲置弧）
# 普通弧（不依赖机组？——路径必须依赖机组，因此 w 以 (p, f->g) 为键）
w = {}
# 普通弧
for (fidx, gidx) in A_links:
    for pidx in crew["pidx"].values:
        w[(pidx, ('F', fidx, gidx))] = pulp.LpVariable(f"w_p{pidx}_F_{fidx}_{gidx}", cat=pulp.LpBinary)
# 源弧 / 汇弧 / S->T
for _, p in crew.iterrows():
    pidx = int(p["pidx"])
    # 源->航班
    for fidx in crew_source_targets[pidx]:
        w[(pidx, ('S', pidx, fidx))] = pulp.LpVariable(f"w_p{pidx}_S_{fidx}", cat=pulp.LpBinary)
    # 航班->汇
    for fidx in crew_sink_sources[pidx]:
        w[(pidx, ('T', pidx, fidx))] = pulp.LpVariable(f"w_p{pidx}_T_{fidx}", cat=pulp.LpBinary)
    # S->T 闲置
    w[(pidx, ('ST', pidx))] = pulp.LpVariable(f"w_p{pidx}_ST", cat=pulp.LpBinary)

# ========== 约束 ==========
# ① 覆盖等式 + 乘机绑定 + 乘机容量
for _, f in flights.iterrows():
    fidx = int(f["idx"])
    reqC = int(f["req_C"])
    reqF = int(f["req_F"])

    prob += pulp.lpSum(xC[(pidx,fidx)] for pidx in crew["pidx"].values) == reqC * z[fidx], f"coverC_f{fidx}"
    prob += pulp.lpSum(xF[(pidx,fidx)] + xFsub[(pidx,fidx)] for pidx in crew["pidx"].values) == reqF * z[fidx], f"coverF_f{fidx}"

    # 乘机-执行绑定与容量随 z 关断
    prob += pulp.lpSum(y[(pidx,fidx)] for pidx in crew["pidx"].values) <= MAX_DEADHEAD_PER_FLIGHT * z[fidx], f"dhCap_f{fidx}"
    for pidx in crew["pidx"].values:
        prob += y[(pidx,fidx)] <= z[fidx], f"y_le_z_p{pidx}_f{fidx}"

# ② 资格域与互斥
for _, p in crew.iterrows():
    pidx = int(p["pidx"])
    c_flag = int(p["Captain"])
    f_flag = int(p["FirstOfficer"])
    d_flag = int(p["Deadhead"])
    for _, f in flights.iterrows():
        fidx = int(f["idx"])
        # 资格绑定
        prob += xC[(pidx,fidx)]    <= c_flag, f"qualC_p{pidx}_f{fidx}"
        prob += xF[(pidx,fidx)]    <= f_flag, f"qualF_p{pidx}_f{fidx}"
        prob += xFsub[(pidx,fidx)] <= (c_flag * f_flag), f"qualFsub_p{pidx}_f{fidx}"
        prob += y[(pidx,fidx)]     <= d_flag, f"qualDH_p{pidx}_f{fidx}"
        # 互斥（同人同航班单形态）
        prob += xC[(pidx,fidx)] + xF[(pidx,fidx)] + xFsub[(pidx,fidx)] + y[(pidx,fidx)] <= 1, f"mutex_p{pidx}_f{fidx}"
        # 节点使用等式
        prob += u[(pidx,fidx)] == (xC[(pidx,fidx)] + xF[(pidx,fidx)] + xFsub[(pidx,fidx)] + y[(pidx,fidx)]), f"u_bind_p{pidx}_f{fidx}"

# ③ 单路径-节点守恒与单位流（基地起止 + S->T 闲置）
# - 对每个机组 p：
#     sum_out(S_p) = 1
#     sum_in(T_p)  = 1
# - 对每个节点 f：
#     sum_in = u = sum_out
for _, p in crew.iterrows():
    pidx = int(p["pidx"])

    # 出源弧：S->f + S->T
    out_S = []
    for fidx in crew_source_targets[pidx]:
        out_S.append(w[(pidx, ('S', pidx, fidx))])
    out_S.append(w[(pidx, ('ST', pidx))])
    prob += pulp.lpSum(out_S) == 1, f"unitflow_outS_p{pidx}"

    # 入汇弧：f->T + S->T （注意：S->T 不“入汇”，它是 S 到 T 的直连；我们计数“入T”的是 f->T）
    in_T = []
    for fidx in crew_sink_sources[pidx]:
        in_T.append(w[(pidx, ('T', pidx, fidx))])
    # 这里不把 ('ST') 计入 in_T，因为它不是某个 f->T 的形式；但对“单位流”已由源侧=1保证闭合
    prob += pulp.lpSum(in_T) == 1, f"unitflow_inT_p{pidx}"

    # 对每个航班节点 f 的入/出守恒 = u
    # 入：S->f（若基地匹配） + ∑(g->f)普通弧
    # 出：f->T（若基地匹配） + ∑(f->h)普通弧
    # 注意：若某机组在 f 上未出现（u=0），入和出均应为 0
    # 入弧集合
    in_edges_by_f = {int(fid): [] for fid in flights["idx"].values}
    out_edges_by_f = {int(fid): [] for fid in flights["idx"].values}

    # 源->f
    for fidx in crew_source_targets[pidx]:
        in_edges_by_f[fidx].append(w[(pidx, ('S', pidx, fidx))])
    # f->T
    for fidx in crew_sink_sources[pidx]:
        out_edges_by_f[fidx].append(w[(pidx, ('T', pidx, fidx))])
    # 普通弧 f->g & g->f
    for (fidx, gidx) in A_links:
        out_edges_by_f[fidx].append(w[(pidx, ('F', fidx, gidx))])
        in_edges_by_f[gidx].append(w[(pidx, ('F', fidx, gidx))])

    for fidx in flights["idx"].values:
        prob += pulp.lpSum(in_edges_by_f[int(fidx)]) == u[(pidx,int(fidx))], f"flow_in_eq_u_p{pidx}_f{fidx}"
        prob += pulp.lpSum(out_edges_by_f[int(fidx)]) == u[(pidx,int(fidx))], f"flow_out_eq_u_p{pidx}_f{fidx}"

# ========== 目标：字典序第 1 层 —— 最大化覆盖 ==========
prob.setObjective(pulp.lpSum(z[fidx] for fidx in flights["idx"].values))

# 求解器设置（优先 CBC）
def solve_lp(lp, msg=True):
    # 优先 CBC，其次 PULP_CBC_CMD 默认
    try:
        solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=None)
        status = lp.solve(solver)
    except Exception:
        status = lp.solve()
    return pulp.LpStatus[status]

t0 = time.time()
status1 = solve_lp(prob, msg=True)
t1 = time.time()
print(f"\n[Layer 1] 覆盖最大化求解状态：{status1}，耗时 {t1-t0:.1f}s")

# 读取最优覆盖值并锁定
Z_star = sum(int(pulp.value(z[fidx])) for fidx in flights["idx"].values)
print(f"[Layer 1] 覆盖的航班数 Z* = {Z_star}")
prob += pulp.lpSum(z[fidx] for fidx in flights["idx"].values) == Z_star, "lock_cover_Zstar"

# ========== 目标：字典序第 2 层 —— 最小化乘机次数 ==========
prob.setObjective(pulp.lpSum(y[(pidx,fidx)] for pidx in crew["pidx"].values for fidx in flights["idx"].values))
t2 = time.time()
status2 = solve_lp(prob, msg=True)
t3 = time.time()
print(f"\n[Layer 2] 乘机最少求解状态：{status2}，耗时 {t3-t2:.1f}s")

Y_star = sum(int(pulp.value(y[(pidx,fidx)])) for pidx in crew["pidx"].values for fidx in flights["idx"].values)
print(f"[Layer 2] 最小乘机次数 Y* = {Y_star}")
prob += pulp.lpSum(y[(pidx,fidx)] for pidx in crew["pidx"].values for fidx in flights["idx"].values) == Y_star, "lock_dh_Ystar"

# ========== 目标：字典序第 3 层 —— 最小化替补次数 ==========
prob.setObjective(pulp.lpSum(xFsub[(pidx,fidx)] for pidx in crew["pidx"].values for fidx in flights["idx"].values))
t4 = time.time()
status3 = solve_lp(prob, msg=True)
t5 = time.time()
print(f"\n[Layer 3] 替补最少求解状态：{status3}，耗时 {t5-t4:.1f}s")

# ========== 结果提取 ==========
# 航班覆盖结果
flights["z"] = flights["idx"].map(lambda i: int(pulp.value(z[int(i)])))
flights["Covered"] = flights["z"].map({1:"已覆盖",0:"未覆盖"})

# 机组-航班 角色/乘机 结果（仅取值为1的）
assign_rows = []
for _, p in crew.iterrows():
    pidx = int(p["pidx"])
    pid  = p["CrewID"]
    base = p["Base"]
    for _, f in flights.iterrows():
        fidx = int(f["idx"])
        role = None
        if int(pulp.value(xC[(pidx,fidx)]))==1:
            role = "正机长"
        elif int(pulp.value(xF[(pidx,fidx)]))==1:
            role = "副机长"
        elif int(pulp.value(xFsub[(pidx,fidx)]))==1:
            role = "正机长替副机长"
        elif int(pulp.value(y[(pidx,fidx)]))==1:
            role = "乘机"
        if role is not None:
            assign_rows.append({
                "CrewID": pid,
                "Base": base,
                "FlightID": f["FlightID"],
                "DepStn": f["DepStn"],
                "ArrStn": f["ArrStn"],
                "DepTime_min": int(f["DepTime_min"]),
                "ArrTime_min": int(f["ArrTime_min"]),
                "Role": role
            })

assign_df = pd.DataFrame(assign_rows).sort_values(["CrewID","DepTime_min","ArrTime_min"]).reset_index(drop=True)

# 未覆盖清单
uncovered = flights.loc[flights["z"]==0, ["FlightID","DepStn","ArrStn","DepTime_min","ArrTime_min","req_C","req_F"]].copy()
uncovered = uncovered.sort_values(["DepTime_min","FlightID"]).reset_index(drop=True)

# ========== 导出结果文件 ==========
uncovered_file = os.path.join(RESULT_DIR, "UncoveredFlights.csv")
roster_file    = os.path.join(RESULT_DIR, "CrewRosters.csv")
flights_file   = os.path.join(RESULT_DIR, "FlightsWithCoverFlag.csv")

uncovered.to_csv(uncovered_file, index=False, encoding="utf-8-sig")
assign_df.to_csv(roster_file, index=False, encoding="utf-8-sig")
flights.to_csv(flights_file, index=False, encoding="utf-8-sig")

print(f"\n已保存结果文件：\n- {uncovered_file}\n- {roster_file}\n- {flights_file}")

# ========== 结果统计与展示 ==========
# 覆盖统计
cover_rate = flights["z"].mean() if len(flights)>0 else 0.0
n_total = len(flights)
n_cov   = flights["z"].sum()
n_unc   = n_total - n_cov
print(f"\n航班总数：{n_total}；已覆盖：{n_cov}；未覆盖：{n_unc}；覆盖率：{cover_rate:.2%}")

# 乘机次数统计（总体与按机组分布）
dh_total = int(assign_df["Role"].eq("乘机").sum())
sub_total = int(assign_df["Role"].eq("正机长替副机长").sum())
print(f"总体乘机次数：{dh_total}；替补使用次数：{sub_total}")

dh_by_crew  = assign_df.query("Role=='乘机'").groupby("CrewID").size().reindex(crew["CrewID"]).fillna(0).astype(int)
sub_by_crew = assign_df.query("Role=='正机长替副机长'").groupby("CrewID").size().reindex(crew["CrewID"]).fillna(0).astype(int)

# 将分钟转回可读时间字符串（用于展示）
def fmt_time(m):
    return pd.to_datetime(int(m), unit="m").strftime("%Y-%m-%d %H:%M")

# 展示前若干结果
print("\n未覆盖航班（前 10 条）：")
print(uncovered.head(10).assign(DepTime=lambda d: d["DepTime_min"].map(fmt_time),
                                ArrTime=lambda d: d["ArrTime_min"].map(fmt_time)
                                ).drop(columns=["DepTime_min","ArrTime_min"]).to_string(index=False))

print("\n机组排班（前 20 条）：")
print(assign_df.head(20).assign(DepTime=lambda d: d["DepTime_min"].map(fmt_time),
                                ArrTime=lambda d: d["ArrTime_min"].map(fmt_time)
                                ).drop(columns=["DepTime_min","ArrTime_min"]).to_string(index=False))

# ========== 可视化：自动生成与展示 ==========
def nice_save(fig, filename):
    if SAVE_PLOTS:
        fig.savefig(os.path.join(RESULT_DIR, filename), dpi=160, bbox_inches="tight")

# 1) 覆盖情况饼图
fig1, ax1 = plt.subplots(figsize=(6,6))
ax1.pie([n_cov, n_unc], labels=["已覆盖","未覆盖"], autopct=lambda p: f"{p:.1f}%",
        startangle=90, pctdistance=0.8, wedgeprops=dict(width=0.45))
ax1.set_title(f"航班覆盖情况（覆盖率 {cover_rate:.1%}）")
nice_save(fig1, "Q1_覆盖情况_饼图.png")
if SHOW_PLOTS: plt.show(); plt.close(fig1)

# 2) 机组乘机次数分布（条形图）
fig2, ax2 = plt.subplots(figsize=(max(6, len(dh_by_crew)*0.4), 4.2))
ax2.bar(dh_by_crew.index.astype(str), dh_by_crew.values)
ax2.set_xlabel("机组")
ax2.set_ylabel("乘机次数")
ax2.set_title("机组乘机次数分布")
for i, v in enumerate(dh_by_crew.values):
    ax2.text(i, v+0.1, str(int(v)), ha="center", va="bottom", fontsize=9, rotation=0)
plt.xticks(rotation=45, ha="right")
nice_save(fig2, "Q1_机组乘机次数分布.png")
if SHOW_PLOTS: plt.show(); plt.close(fig2)

# 3) 机组替补次数分布（条形图）
fig3, ax3 = plt.subplots(figsize=(max(6, len(sub_by_crew)*0.4), 4.2))
ax3.bar(sub_by_crew.index.astype(str), sub_by_crew.values)
ax3.set_xlabel("机组")
ax3.set_ylabel("替补次数")
ax3.set_title("机组正机长替副机长次数分布")
for i, v in enumerate(sub_by_crew.values):
    ax3.text(i, v+0.1, str(int(v)), ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=45, ha="right")
nice_save(fig3, "Q1_机组替补次数分布.png")
if SHOW_PLOTS: plt.show(); plt.close(fig3)

# 4) 不同机场未覆盖量（若存在未覆盖）
if n_unc > 0:
    unc_by_stn = uncovered.groupby(["DepStn"]).size().sort_values(ascending=False)
    fig4, ax4 = plt.subplots(figsize=(max(6, len(unc_by_stn)*0.45), 4.2))
    ax4.bar(unc_by_stn.index.astype(str), unc_by_stn.values)
    ax4.set_xlabel("出发机场")
    ax4.set_ylabel("未覆盖航班数")
    ax4.set_title("各出发机场未覆盖航班分布")
    for i, v in enumerate(unc_by_stn.values):
        ax4.text(i, v+0.1, str(int(v)), ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    nice_save(fig4, "Q1_未覆盖航班_按出发机场.png")
    if SHOW_PLOTS: plt.show(); plt.close(fig4)

# 5) 按时间轴的覆盖/未覆盖（可选：采样提示）
#   为避免 B 套图像过密，这里对时间做等距分箱展示
bins = 24  # 分箱数
tmin, tmax = flights["DepTime_min"].min(), flights["DepTime_min"].max()
flights["_bin"] = pd.cut(flights["DepTime_min"], bins=bins, labels=False, include_lowest=True)
cov_by_bin = flights.groupby(["_bin","z"]).size().unstack(fill_value=0)
cov_by_bin = cov_by_bin.rename(columns={0:"未覆盖",1:"已覆盖"})
fig5, ax5 = plt.subplots(figsize=(10,4))
cov_by_bin.plot(kind="bar", stacked=True, ax=ax5, width=0.8)
ax5.set_xlabel("起飞时间分箱（等宽）")
ax5.set_ylabel("航班数量")
ax5.set_title("按时间分箱的覆盖/未覆盖分布")
plt.xticks(rotation=0)
nice_save(fig5, "Q1_时间分箱_覆盖分布.png")
if SHOW_PLOTS: plt.show(); plt.close(fig5)

print("\n--- 运行完成：结果文件与图片已生成并显示 ---")
