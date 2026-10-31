#!/usr/bin/env python3
"""本机 GPU 测试：单卡带宽、卡与卡之间的带宽和时延（按 PIX/PXB/PHB/NODE/SYS 汇总）、8 卡 NCCL all-reduce。

由 run_gpu_test.sh 运行（带上与部署一致的 NCCL 环境变量，报告写入 logs/），也可以直接 python3 gpu_local_test.py。
只测本机，两台各跑一次，不需要同时运行。服务在跑时不要测：会抢 GPU，结果也不准。

报告分五部分：
  1. 单卡：主机内存 ↔ GPU 的拷贝带宽（主机内存取自卡所在的 NUMA 节点）、显存拷贝带宽、PCIe 链路
  2. 卡与卡之间的连接类型（nvidia-smi topo -m）和能否 P2P
  3. 每对卡的单向、双向拷贝带宽和 16 字节拷贝的时延（与 CUDA 示例 p2pBandwidthLatencyTest 同口径），并校验数据
  4. 按连接类型汇总，列出比同类明显慢的卡对
  5. NCCL all-reduce：decode 一步、prefill 一块大小的消息，即 TP8 实际用的通信

1–4 在本进程里直接调 CUDA driver API（libcuda）做拷贝，用 GPU 事件计时；5 每张卡起一个进程。
有 FAIL 项时以退出码 1 结束。
"""
import contextlib
import ctypes
import datetime
import json
import mmap
import os
import re
import socket
import statistics
import subprocess
import sys
import time
import traceback
import unicodedata

# 下面的环境变量要在 import torch 之前设置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 卡号与 nvidia-smi 一致
for _k in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
    os.environ.pop(_k, None)  # 用默认显存分配器：expandable_segments 分出的显存要另外授权才能跨卡访问

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MiB, GiB = 1 << 20, 1 << 30
SIZE, ITERS = 256 * MiB, 10       # 带宽：每次拷贝的字节数、次数
LAT_BYTES, LAT_ITERS = 16, 100    # 时延：每次拷贝的字节数、次数
SPIN = 50_000_000                 # 计时前让 GPU 空转的时钟周期（约 20 ms），留给 CPU 把拷贝全部排进队列
GEN = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}  # PCIe 速率（GT/s）→ 第几代
TOPO = {  # nvidia-smi topo -m 的连接类型，由近到远
    "PIX": "最多经过一个 PCIe 桥（同一个交换芯片下）",
    "PXB": "经过多个 PCIe 桥，不经过 CPU",
    "PHB": "经过 CPU 的 PCIe 主桥",
    "NODE": "同一 NUMA 节点内，跨 PCIe 主桥",
    "SYS": "跨 NUMA 节点，经过 CPU 之间的互连（UPI / xGMI）",
}
ALL_CPUS = os.sched_getaffinity(0)
P2P_OFF = os.environ.get("NCCL_P2P_DISABLE") == "1"  # cluster.env 的 GPU_P2P=0（setup_nccl_env 据此设置）
STREAMS = []   # 每张卡一条测试流
RESULTS = []   # 每条结论的级别，最后统计

# ---------------- CUDA driver API ----------------
_CU_ARGS = {
    "cuInit": (ctypes.c_uint,),
    "cuDeviceGet": (ctypes.POINTER(ctypes.c_int), ctypes.c_int),
    "cuDeviceGetPCIBusId": (ctypes.c_char_p, ctypes.c_int, ctypes.c_int),
    "cuDeviceCanAccessPeer": (ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int),
    "cuDevicePrimaryCtxRetain": (ctypes.POINTER(ctypes.c_void_p), ctypes.c_int),
    "cuCtxEnablePeerAccess": (ctypes.c_void_p, ctypes.c_uint),
    "cuMemHostRegister_v2": (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint),
    "cuMemHostUnregister": (ctypes.c_void_p,),
    "cuMemcpyAsync": (ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p),
    "cuGetErrorString": (ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)),
}
_cu = {}


def cu(name, *args, check=True):
    """调用 CUDA driver API，返回 CUresult；check 时出错就抛异常。"""
    if not _cu:
        lib = ctypes.CDLL("libcuda.so.1")
        for k, argtypes in _CU_ARGS.items():
            f = getattr(lib, k)
            f.argtypes, f.restype = argtypes, ctypes.c_int
            _cu[k] = f
    rc = _cu[name](*args)
    if check and rc:
        msg = ctypes.c_char_p()
        _cu["cuGetErrorString"](rc, ctypes.byref(msg))
        raise RuntimeError(f"{name} 失败：{(msg.value or b'').decode()}（CUresult {rc}）")
    return rc


# ---------------- 输出 ----------------
def say(level, msg, *hints):
    """打印一条结论（OK / WARN / FAIL，计入最后的统计），hints 是下面缩进的说明。"""
    RESULTS.append(level)
    print(f"[{level:<4}] {msg}")
    for h in hints:
        print(f"       {h}")


def section(title):
    print(f"\n== {title} ==")


def dw(s):  # 终端显示宽度：中文占两格
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def table(rows, right=()):
    """按显示宽度对齐打印，right 里的列右对齐。"""
    widths = [max(dw(r[k]) for r in rows) for k in range(len(rows[0]))]
    for r in rows:
        cells = [" " * (w - dw(c)) + c if k in right else c + " " * (w - dw(c))
                 for k, (c, w) in enumerate(zip(r, widths))]
        print("  ".join(cells).rstrip())


def matrix(m, can):
    """卡与卡之间的数值矩阵，行是发送方、列是接收方；不能 P2P 的标 *。"""
    n = len(m)
    rows = [[""] + [f"GPU{j}" for j in range(n)]]
    for i in range(n):
        rows.append([f"GPU{i}"] + ["-" if i == j else f"{m[i][j]:.1f}" + ("" if can[i][j] else "*")
                                   for j in range(n)])
    table(rows, right=range(1, n + 1))


def grade(v, good, okay):
    return "OK" if v >= good else ("WARN" if v >= okay else "FAIL")


def fmt_bytes(b):
    return f"{b / MiB:.0f} MiB" if b >= MiB else f"{b / 1024:.0f} KiB"


# ---------------- 硬件信息 ----------------
def sh(*cmd):
    """运行命令，返回标准输出；命令不存在或出错时返回空串。"""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def norm_bdf(s):  # 统一成 sysfs 的写法，如 0000:18:00.0
    dom, bus, rest = s.strip().lower().split(":")
    return f"{int(dom, 16):04x}:{bus}:{rest}"


def link(dev, kind):
    """sysfs 里 PCI 设备的链路（速率 GT/s, 宽度），kind 是 max 或 current；读不到时为 None。"""
    try:
        return float(read(f"{dev}/{kind}_link_speed").split()[0]), int(read(f"{dev}/{kind}_link_width"))
    except (ValueError, IndexError):
        return None


def fmt_link(lk):
    return f"Gen{GEN.get(lk[0], '?')} x{lk[1]}" if lk else "?"


def link_gbps(lk):  # 链路的理论单向带宽（GB/s）；Gen3 起是 128b/130b 编码
    return lk[0] * lk[1] * (128 / 130 if lk[0] >= 8 else 0.8) / 8


def gpu_info(i):
    """第 i 张卡的 PCI 地址、NUMA 节点、本地 CPU，以及按卡和上游端口的能力算出的应有链路。"""
    dev, bus = ctypes.c_int(), ctypes.create_string_buffer(32)
    cu("cuDeviceGet", ctypes.byref(dev), i)
    cu("cuDeviceGetPCIBusId", bus, 32, dev)
    bdf = norm_bdf(bus.value.decode())
    path = os.path.realpath(f"/sys/bus/pci/devices/{bdf}")
    mine, up = link(path, "max"), link(os.path.dirname(path), "max")  # 上游是 CPU 根端口或交换芯片的下行口
    want = (min(mine[0], up[0]), min(mine[1], up[1])) if mine and up else mine
    cpus = set()
    for part in filter(None, read(f"{path}/local_cpulist").split(",")):
        a, _, b = part.partition("-")
        cpus.update(range(int(a), int(b or a) + 1))
    numa = read(f"{path}/numa_node")
    return {"dev": dev.value, "bdf": bdf, "path": path, "want": want,
            "numa": numa if numa not in ("", "-1") else "-", "cpus": cpus & ALL_CPUS}


def topo_matrix(gpus):
    """nvidia-smi topo -m 里两两之间的连接类型，按本进程的卡号排；拿不到时是 ?。"""
    smi = {}  # PCI 地址 → nvidia-smi 的卡号
    for line in sh("nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader").splitlines():
        m = re.match(r"\s*(\d+)\s*,\s*([0-9a-fA-F]+:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])", line)
        if m:
            smi[norm_bdf(m.group(2))] = int(m.group(1))
    rows = {}
    for line in re.sub(r"\x1b\[[0-9;]*m", "", sh("nvidia-smi", "topo", "-m")).splitlines():
        cells = [c.strip() for c in line.split("\t")]
        if re.fullmatch(r"GPU\d+", cells[0]):
            rows[int(cells[0][3:])] = cells[1:]
    n = len(gpus)
    out = [["?"] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            row, k = rows.get(smi.get(gpus[i]["bdf"])), smi.get(gpus[j]["bdf"])
            if row and k is not None and k < len(row) and row[k]:
                out[i][j] = row[k]
    return out


def topo_rank(cls):  # 连接类型由近到远排序：NVLink（NV#）最近，不认识的排最后
    order = list(TOPO)
    return -1 if cls.startswith("NV") else order.index(cls) if cls in order else len(order)


# ---------------- 拷贝与计时 ----------------
@contextlib.contextmanager
def host_buffer(cpus):
    """SIZE 字节的锁页主机内存，放在 cpus 所在的 NUMA 节点上，返回地址。"""
    if cpus:
        os.sched_setaffinity(0, cpus)  # 内存页落在第一次写它的 CPU 所在的 NUMA 节点
    m = mmap.mmap(-1, SIZE)
    view = ctypes.c_char.from_buffer(m)
    addr = ctypes.addressof(view)
    try:
        ctypes.memset(addr, 0, SIZE)
        cu("cuMemHostRegister_v2", addr, SIZE, 0x3)  # PORTABLE | DEVICEMAP：每张卡都能直接 DMA
        try:
            yield addr
        finally:
            cu("cuMemHostUnregister", addr)
    finally:
        del view
        m.close()
        os.sched_setaffinity(0, ALL_CPUS)


def run(jobs, n, spin=True):
    """jobs 里每项是（卡号, 目标地址, 源地址, 字节数），各自在本卡的测试流上同时开始，各拷贝 n 次。
    返回从开始到全部拷完的 GPU 时间（秒）。"""
    first = STREAMS[jobs[0][0]]
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    if spin:  # 先让 GPU 空转，CPU 把拷贝全部排进队列后再一起开始：计时不含 CPU 的提交开销
        with torch.cuda.stream(first):
            torch.cuda._sleep(SPIN)
    start.record(first)
    for d, *_ in jobs[1:]:
        STREAMS[d].wait_event(start)
    for _ in range(n):
        for d, dst, src, nbytes in jobs:
            torch.cuda.set_device(d)
            cu("cuMemcpyAsync", dst, src, nbytes, STREAMS[d].cuda_stream)
    for d, *_ in jobs[1:]:
        done = torch.cuda.Event()
        done.record(STREAMS[d])
        first.wait_event(done)
    stop.record(first)
    deadline = time.time() + 60
    while not stop.query():  # 不用 synchronize：P2P 卡死时能报错退出，而不是一直挂着
        if time.time() > deadline:
            raise TimeoutError("、".join(f"GPU{d}" for d, *_ in jobs) + " 上的拷贝 60 秒没有完成（P2P 卡死？）")
        time.sleep(0.001)
    return start.elapsed_time(stop) / 1e3


def measure(jobs, n=ITERS):
    run(jobs, 2, spin=False)  # 预热：第一次跨卡拷贝要建立映射，GPU 也要从空闲状态升频
    return run(jobs, n)


def gbps(jobs, n=ITERS):
    """jobs 同时各拷贝 n 次的合计带宽（GB/s）。"""
    return sum(j[3] for j in jobs) * n / measure(jobs, n) / 1e9


# ---------------- 1. 单卡 ----------------
def test_single(gpus, S, R):
    section(f"1. 单卡（GB/s；主机内存取自卡所在的 NUMA 节点；每项 {SIZE // MiB} MiB × {ITERS} 次）")
    rows = [["卡", "PCI 地址", "NUMA", "PCIe 应为", "实测链路", "主机→GPU", "GPU→主机", "显存拷贝"]]
    h2d, d2h, mem, cur = [], [], [], []
    for i, g in enumerate(gpus):
        torch.cuda.set_device(i)
        with host_buffer(g["cpus"]) as h:
            h2d.append(gbps([(i, R[i], h, SIZE)]))
            cur.append(link(g["path"], "current"))  # 刚测完，GPU 还没回到空闲状态
            d2h.append(gbps([(i, h, S[i], SIZE)]))
        mem.append(2 * gbps([(i, R[i], S[i], SIZE)]))  # 显存拷贝同时读和写，按两倍算
        rows.append([f"GPU{i}", g["bdf"], g["numa"], fmt_link(g["want"]), fmt_link(cur[i]),
                     f"{h2d[i]:.1f}", f"{d2h[i]:.1f}", f"{mem[i]:.0f}"])
    table(rows, right={2, 5, 6, 7})
    print("（PCIe 应为：卡和上游端口都支持的最高规格；实测链路：刚测完主机→GPU 时读到的。"
          "GPU 空闲时会自动降低 PCIe 速率，平时看到低代数是正常的）")

    lows = [min(a, b) for a, b in zip(h2d, d2h)]
    med = statistics.median(lows)
    bad = 0
    for i, g in enumerate(gpus):
        lo, want, c = lows[i], g["want"], cur[i]
        hints = [] if g["cpus"] else ["没能把主机内存放到卡所在的 NUMA 节点（读不到 local_cpulist），结果可能偏低"]
        if want and lo < 0.7 * link_gbps(want):
            if c and (c[0] < want[0] or c[1] < want[1]):
                hints.append(f"链路只跑到 {fmt_link(c)}：检查插槽、转接线和 BIOS 里的 PCIe 速率设置")
            else:
                hints.append("链路正常但带宽低：检查 IOMMU 是否为 passthrough（sudo ./host_setup.sh 查看）")
            say("WARN", f"GPU{i} 主机↔GPU 只有 {lo:.1f} GB/s，是 {fmt_link(want)} 理论值 "
                        f"{link_gbps(want):.0f} GB/s 的 {lo / link_gbps(want):.0%}", *hints)
            bad += 1
        elif not want and lo < 0.7 * med:
            say("WARN", f"GPU{i} 主机↔GPU 只有 {lo:.1f} GB/s，明显低于其它卡（中位数 {med:.1f} GB/s）",
                *hints, "检查这张卡的插槽和 PCIe 链路")
            bad += 1
    if not bad:
        i = lows.index(min(lows))
        want = gpus[i]["want"]
        ref = (f"，是 {fmt_link(want)} 理论值 {link_gbps(want):.0f} GB/s 的 {lows[i] / link_gbps(want):.0%}"
               if want else "（读不到 PCIe 链路信息，没有和理论值比较）")
        say("OK", f"主机↔GPU 带宽最低 {lows[i]:.1f} GB/s（GPU{i}）{ref}")

    med = statistics.median(mem)
    slow = [i for i in range(len(gpus)) if mem[i] < 0.85 * med]
    for i in slow:
        say("WARN", f"GPU{i} 显存拷贝 {mem[i]:.0f} GB/s，比各卡中位数 {med:.0f} GB/s 低 {1 - mem[i] / med:.0%}",
            "可能在降频：nvidia-smi -q -d PERFORMANCE 查看是否触发了温度或功耗限制")
    if not slow:
        say("OK", f"显存拷贝 {min(mem):.0f}–{max(mem):.0f} GB/s，各卡一致")
    return h2d


# ---------------- 2. 连接类型与 P2P ----------------
def test_topology(gpus):
    n = len(gpus)
    section("2. 卡与卡之间的连接（nvidia-smi topo -m）")
    topo = topo_matrix(gpus)
    names = [f"GPU{i}" for i in range(n)]
    table([[""] + names] + [[names[i]] + ["X" if i == j else topo[i][j] for j in range(n)] for i in range(n)],
          right=range(1, n + 1))
    kinds = {topo[i][j] for i in range(n) for j in range(n) if i != j}
    for cls in sorted(kinds, key=topo_rank):
        if cls in TOPO:
            print(f"  {cls}：{TOPO[cls]}")
    if "?" in kinds:
        say("WARN", "读不到 nvidia-smi topo -m，第 4 部分没法按连接类型汇总")

    ctx = []
    for g in gpus:
        c = ctypes.c_void_p()
        cu("cuDevicePrimaryCtxRetain", ctypes.byref(c), g["dev"])
        ctx.append(c)
    can = [[True] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ok = ctypes.c_int()
            cu("cuDeviceCanAccessPeer", ctypes.byref(ok), gpus[i]["dev"], gpus[j]["dev"])
            torch.cuda.set_device(i)  # 让第 i 张卡的上下文成为当前上下文，下面给它开通访问第 j 张卡的权限
            # 704：已经开通过
            can[i][j] = bool(ok.value) and cu("cuCtxEnablePeerAccess", ctx[j], 0, check=False) in (0, 704)
    no = [f"GPU{i}↔GPU{j}" for i in range(n) for j in range(i + 1, n) if not (can[i][j] and can[j][i])]
    if no:
        say("WARN", "这些卡对不能 P2P，拷贝要经主机内存中转（NCCL 会改走主机共享内存，SGLang 的自定义 all-reduce 也要求 P2P）：",
            "、".join(no), "检查 BIOS 里的 ACS / IOMMU（sudo ./host_setup.sh 查看）；有些平台本来就不支持跨 CPU 的 P2P")
    else:
        say("OK", "驱动报告所有卡对都能 P2P（数据对不对看第 3 部分的校验）")
    return topo, can


# ---------------- 3. 每对卡 ----------------
def test_pairs(can, send, recv, S, R):
    n = len(can)
    section("3. 卡与卡之间的带宽和时延")
    print(f"[INFO] 带宽每项 {SIZE // MiB} MiB × {ITERS} 次；时延是 {LAT_BYTES} 字节拷贝的单次耗时，"
          f"与 CUDA 示例 p2pBandwidthLatencyTest 同口径")
    print(f"[INFO] 正在测 {n * (n - 1)} 个方向的单向带宽和时延、{n * (n - 1) // 2} 对卡的双向带宽……")
    sums = [t.sum().item() for t in send]  # 每张卡发送的数据都不同，拷完比对接收方的校验和
    uni = [[0.0] * n for _ in range(n)]
    bi = [[0.0] * n for _ in range(n)]
    lat = [[0.0] * n for _ in range(n)]
    broken = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            uni[i][j] = gbps([(i, R[j], S[i], SIZE)])
            if recv[j].sum().item() != sums[i]:
                broken.append(f"GPU{i}→GPU{j}")
            lat[i][j] = measure([(i, R[j], S[i], LAT_BYTES)], LAT_ITERS) / LAT_ITERS * 1e6
    for i in range(n):
        for j in range(i + 1, n):
            bi[i][j] = bi[j][i] = gbps([(i, R[j], S[i], SIZE), (j, R[i], S[j], SIZE)])

    print("\n单向带宽（GB/s，行发列收）")
    matrix(uni, can)
    print("\n双向带宽（GB/s，两个方向同时拷贝的合计）")
    matrix(bi, can)
    print("\n时延（µs，行发列收）")
    matrix(lat, can)
    if not all(map(all, can)):
        print("* 不能 P2P，经主机内存中转")
    print()
    which = f"全部 {len(broken)} 个方向" if len(broken) == n * (n - 1) else "、".join(broken)
    if broken and P2P_OFF:
        say("WARN", f"P2P 拷贝后的数据不对：{which}",
            "已按 GPU_P2P=0 关掉 P2P：NCCL 机内走主机共享内存，SGLang 不用自定义 all-reduce，部署不受影响")
    elif broken:
        say("FAIL", f"P2P 拷贝后的数据不对：{which}",
            "检查 BIOS 里的 ACS / IOMMU（sudo ./host_setup.sh 查看）；修不好就在 cluster.env 设 GPU_P2P=0 不用 P2P",
            "带宽还超过了 PCIe 链路的上限时，多半是装了 P2P 补丁驱动的消费级卡（BAR1 小于显存时写不对），只能设 GPU_P2P=0",
            "修好或关掉之前，NCCL 可能卡住或算错")
    else:
        say("OK", "所有方向拷贝后的数据都校验一致")
    return uni, bi, lat, broken


# ---------------- 4. 按连接类型汇总 ----------------
def summarize(topo, uni, bi, lat, h2d):
    n = len(topo)
    section("4. 按连接类型汇总")
    groups = {}
    for i in range(n):
        for j in range(i + 1, n):
            groups.setdefault(topo[i][j], []).append((i, j))
    rows = [["类型", "卡对", "单向 最低/中位", "双向 最低/中位", "时延 中位/最高"]]
    med, slow = {}, []
    for cls in sorted(groups, key=topo_rank):
        prs = groups[cls]
        u = [v for i, j in prs for v in (uni[i][j], uni[j][i])]
        b = [bi[i][j] for i, j in prs]
        lt = [v for i, j in prs for v in (lat[i][j], lat[j][i])]
        mu, mb, ml = statistics.median(u), statistics.median(b), statistics.median(lt)
        med[cls] = mu
        rows.append([cls, str(len(prs)), f"{min(u):.1f} / {mu:.1f}", f"{min(b):.1f} / {mb:.1f}",
                     f"{ml:.1f} / {max(lt):.1f}"])
        for i, j in prs:
            pu, pb, pl = min(uni[i][j], uni[j][i]), bi[i][j], max(lat[i][j], lat[j][i])
            if pu < 0.7 * mu or pb < 0.7 * mb or pl > 1.5 * ml:
                slow.append(f"GPU{i}↔GPU{j}（{cls}）：单向 {pu:.1f} GB/s，双向 {pb:.1f} GB/s，时延 {pl:.1f} µs")
    table(rows, right={1, 2, 3, 4})
    print("（带宽单位 GB/s，时延单位 µs）")
    if slow:
        say("WARN", f"{len(slow)} 对卡明显比同类慢（带宽低于同类中位数的 70%，或时延超过同类中位数的 1.5 倍）：", *slow)
    else:
        say("OK", "同类卡对之间的带宽和时延一致")

    ref = statistics.median(h2d)
    weak = [c for c in med if med[c] < 0.5 * ref]
    for c in weak:
        hints = [f"{c}：{TOPO[c]}"] if c in TOPO else []
        if c == "SYS":
            hints.append("TP8 的 all-reduce 一定会经过跨 CPU 的卡对。可以用 NCCL_P2P_LEVEL=NODE ./run_gpu_test.sh "
                         "让这些卡对改走主机内存，对比第 5 部分的结果")
        say("WARN", f"{c} 卡对的单向带宽中位数 {med[c]:.1f} GB/s，不到单卡主机→GPU 带宽（{ref:.1f} GB/s）的一半",
            *hints)
    if not weak:
        say("OK", f"各类卡对的单向带宽都在单卡主机→GPU 带宽（{ref:.1f} GB/s）的一半以上")


# ---------------- 5. NCCL all-reduce ----------------
def hidden_size():
    try:
        with open(os.path.join(os.environ.get("MODEL_PATH", ""), "config.json")) as f:
            cfg = json.load(f)
        return int(cfg.get("text_config", cfg)["hidden_size"]), "来自 config.json"
    except Exception:
        return 6144, "假设值，没读到 config.json"


def allreduce_worker(rank, world, port, sizes, q):
    """第 5 部分：每张卡一个进程（torch.multiprocessing.spawn 启动），结果由 rank 0 放进队列。"""
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world,
                            timeout=datetime.timedelta(minutes=3))
    out, bad = [], torch.zeros(1, dtype=torch.int32, device="cuda")
    for nbytes, iters in sizes:
        # 先校验结果：每张卡填 rank+1，和应为 1+2+…+world（BF16 能精确表示）
        x = torch.full((nbytes // 2,), rank + 1, dtype=torch.bfloat16, device="cuda")
        dist.all_reduce(x)
        bad += int(not bool((x == world * (world + 1) // 2).all()))
        x.zero_()
        for _ in range(5):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(SPIN)  # 同第 3 部分：all-reduce 全部排好队再一起开始，计时不含 CPU 的下发开销
        start.record()
        works = [dist.all_reduce(x, async_op=True) for _ in range(iters)]
        for w in works:
            w.wait()
        stop.record()
        stop.synchronize()
        t = torch.tensor([start.elapsed_time(stop) / 1e3 / iters], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        out.append(t.item())
    dist.all_reduce(bad)
    if rank == 0:
        q.put((out, bad.item()))
    dist.destroy_process_group()


def test_allreduce(n):
    H, src = hidden_size()
    batch = int(os.environ.get("MAX_RUNNING") or 32)
    chunk = int(os.environ.get("CHUNKED_PREFILL") or 16384)
    keys = ("NCCL_P2P_LEVEL", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_MIN_NCHANNELS", "NCCL_ALGO", "NCCL_PROTO")
    env = " ".join(f"{k}={os.environ[k]}" for k in keys if os.environ.get(k)) or "NCCL 默认设置"
    section(f"5. {n} 卡 NCCL all-reduce（TP{n} 实际用的通信；{env}）")
    sizes = [(batch * H * 2, 100), (chunk * H * 2, 10)]  # BF16 hidden：decode 一步、prefill 一块
    print(f"[INFO] 每张卡起一个进程测，消息大小按 hidden={H} 算（{src}）")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    q = mp.get_context("spawn").SimpleQueue()
    try:
        mp.spawn(allreduce_worker, args=(n, port, sizes, q), nprocs=n)
        (t_dec, t_pre), bad = q.get()
    except Exception as e:  # 子进程出错或被 NCCL 超时终止
        say("FAIL", "NCCL all-reduce 出错：", *str(e).strip().splitlines()[-20:],
            "单机 NCCL 也会初始化网卡：NCCL_SOCKET_IFNAME / NCCL_IB_HCA 指的网卡不可用时这里也会失败；"
            "网卡没问题就看第 2、3 部分的 P2P 结果")
        return
    if bad:
        say("FAIL", f"all-reduce 的结果不对（{bad} 张卡次）：NCCL 机内传输把数据写错了",
            "P2P 拷贝的数据也不对时在 cluster.env 设 GPU_P2P=0；否则检查 BIOS 里的 ACS / IOMMU")
    else:
        say("OK", "all-reduce 的结果校验正确")
    print(f"[INFO] decode 一步（{batch} 个请求，{fmt_bytes(sizes[0][0])}）：{t_dec * 1e6:.0f} µs/次")
    bw = sizes[1][0] * 2 * (n - 1) / n / t_pre / 1e9
    g = grade(bw, 20, 8)
    hints = [] if g == "OK" else ["先看第 3、4 部分有没有慢的卡对；也可以换 NCCL 设置对比，如 NCCL_P2P_LEVEL=NODE ./run_gpu_test.sh"]
    say(g, f"prefill 一块（{chunk} token，{fmt_bytes(sizes[1][0])}）：{t_pre * 1e3:.1f} ms/次，"
           f"总线带宽 {bw:.1f} GB/s（20 以上正常）", *hints)


# ---------------- 主流程 ----------------
def header(gpus):
    n = len(gpus)
    names = " / ".join(sorted({torch.cuda.get_device_name(i) for i in range(n)}))
    drv = (sh("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").split() or ["?"])[0]
    nv = torch.cuda.nccl.version()
    nv = ".".join(map(str, nv)) if isinstance(nv, tuple) else str(nv)
    print("=" * 78)
    print(f"本机 GPU 测试 · {socket.gethostname()} · {n} × {names}")
    print(f"驱动 {drv} · torch {torch.__version__} · CUDA {torch.version.cuda} · NCCL {nv}")
    print("=" * 78)


def main():
    if not torch.cuda.is_available():
        say("FAIL", "torch 看不到 GPU（容器需要 --gpus all）")
        return
    cu("cuInit", 0)
    n = torch.cuda.device_count()
    gpus = [gpu_info(i) for i in range(n)]
    header(gpus)
    if n != 8:
        say("WARN", f"看到 {n} 张卡，本方案每台要 8 张")
    busy = []
    for i in range(n):
        free, total = torch.cuda.mem_get_info(i)
        if total - free > 2 * GiB:
            busy.append(f"GPU{i} {(total - free) / GiB:.0f} GiB")
    if busy:
        say("WARN", "这些卡上已经有显存被占用（服务还在跑？），结果会偏低：" + "、".join(busy))

    STREAMS.extend(torch.cuda.Stream(device=i) for i in range(n))
    send, recv = [], []
    for i in range(n):
        g = torch.Generator(device=f"cuda:{i}")
        g.manual_seed(i)  # 每张卡发送的数据不同，P2P 写丢了才校验得出来
        send.append(torch.randint(0, 256, (SIZE,), dtype=torch.uint8, device=f"cuda:{i}", generator=g))
        recv.append(torch.empty(SIZE, dtype=torch.uint8, device=f"cuda:{i}"))
    S, R = [t.data_ptr() for t in send], [t.data_ptr() for t in recv]

    h2d = test_single(gpus, S, R)
    if n < 2:
        return
    topo, can = test_topology(gpus)
    uni, bi, lat, broken = test_pairs(can, send, recv, S, R)
    summarize(topo, uni, bi, lat, h2d)
    del send, recv
    torch.cuda.empty_cache()
    if broken and not P2P_OFF:
        print("\n[INFO] 卡间拷贝的数据不对，跳过 all-reduce（NCCL 走 P2P 大概率会卡住）；设 GPU_P2P=0 后再测")
    else:
        test_allreduce(n)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # 经 tee 写日志时也能边测边看
    aborted = False
    try:
        main()
    except Exception:  # CUDA 出错或拷贝卡住后，本进程里的 GPU 状态已经不可信，不再往下测
        traceback.print_exc()
        say("FAIL", "测试中途出错，见上面的报错")
        aborted = True
    nfail, nwarn = RESULTS.count("FAIL"), RESULTS.count("WARN")
    print("\n" + "=" * 78)
    print(f"结论：{nfail} 项 FAIL，{nwarn} 项 WARN")
    print("=" * 78)
    sys.stdout.flush()
    sys.stderr.flush()
    if aborted:
        os._exit(1)  # 正常退出可能卡在 CUDA 的清理上
    sys.exit(1 if nfail else 0)
