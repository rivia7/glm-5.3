#!/usr/bin/env python3
"""跨机 NCCL 自检：按 PP2×TP8 的通信形态测带宽和时延。

由 run_nccl_test.sh 用 torchrun 在两台机器上各起 8 个进程，不要直接运行。
rank 0-7 在节点0，rank 8-15 在节点1；rank i 与 rank i+8 是流水级边界上的一对。

先校验数据，再测四项：
  0. 数据校验：机内 all-reduce 的结果、跨机两个方向 send/recv 收到的内容。
     P2P 或 GPUDirect RDMA 把数据写错时 NCCL 不会报错，只能这样查出来
  1. 机内 8 卡 all-reduce（TP 组内，走 PCIe）
  2. 跨机单对 send/recv（rank 0 -> rank 8）
  3. 跨机 8 对同时 send/recv（PP 边界的真实形态：8 对共用一块网卡）
  4. 跨机 1 MiB 往返时延（约等于 decode 一步的边界传输量）

有 FAIL 项时，两台机器上的所有进程都以退出码 1 结束。
"""
import collections
import datetime
import json
import os
import socket
import sys
import time

import torch
import torch.distributed as dist

G = 8                                                  # 每台机器的 GPU 数
CPU_MODE = os.environ.get("NCCL_TEST_CPU") == "1"      # 仅用于在没有 GPU 的机器上检查脚本逻辑
SCALE = 64 if CPU_MODE else 1
LINE = float(os.environ.get("LINE_RATE_GBPS", "400")) / 8   # 400 Gb/s ≈ 50 GB/s
MiB = 1 << 20
# 通信卡住时多久报错退出（默认 10 分钟太久）。P2P 不通时 NCCL 往往不报错，而是一直等
TIMEOUT = datetime.timedelta(seconds=int(os.environ.get("NCCL_TEST_TIMEOUT", "300")))


def sync():
    if not CPU_MODE:
        torch.cuda.synchronize()


def timed(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


def grade(v, good, okay, higher_is_better=True):
    if not higher_is_better:
        v, good, okay = -v, -good, -okay
    return "OK  " if v >= good else ("WARN" if v >= okay else "FAIL")


def hidden_size():
    path = os.path.join(os.environ.get("MODEL_PATH", ""), "config.json")
    try:
        cfg = json.load(open(path))
        return int(cfg.get("text_config", cfg)["hidden_size"]), "来自 config.json"
    except Exception:
        return 6144, "假设值，未读到 config.json"


def verify(rank, node, peer, tp, dev):
    """数据校验，返回出错项的说明。小消息和大消息在 NCCL 里走不同的协议，两种都要对。"""
    errs = []
    sizes = ((4 << 10, "4 KiB"), (64 * MiB, "64 MiB"))
    # 机内 all-reduce：每个 rank 填 local+1，结果应是 1+2+…+8
    want = G * (G + 1) // 2
    for nbytes, label in sizes:
        x = torch.full((max(1, nbytes // SCALE // 4),), rank % G + 1, dtype=torch.int32, device=dev)
        dist.all_reduce(x, group=tp)
        if not bool((x == want).all()):
            errs.append(f"机内 all-reduce（{label}）结果不对")
    # 跨机 send/recv：节点0→节点1、节点1→节点0 各发一次，内容随发送方不同
    for nbytes, label in sizes:
        base = torch.arange(max(1, nbytes // SCALE // 4), dtype=torch.int32, device=dev)
        for src in (0, 1):
            if node == src:
                dist.send(base + rank * 7919, dst=peer)
            else:
                y = torch.empty_like(base)
                dist.recv(y, src=peer)
                if not torch.equal(y, base + peer * 7919):
                    errs.append(f"跨机 send/recv（节点{src}→节点{1 - src}，{label}）收到的数据不对")
    sync()
    return errs


def report(r):
    bad = collections.Counter(e for x in r for e in x["errs"])
    ar = [min(x["ar_busbw"] for x in r[n * G:(n + 1) * G]) for n in range(2)]
    pair_t = max(r[0]["pair_t"], r[G]["pair_t"])
    pair_bw = r[G]["pair_nb"] / pair_t / 1e9
    all8_t = max(x["all8_t"] for x in r)
    all8_bw = G * r[G]["all8_nb"] / all8_t / 1e9
    one_way_us = r[0]["rtt"] / 2 * 1e6

    print("\n" + "=" * 72)
    print(f"节点0：{r[0]['host']}    节点1：{r[G]['host']}    网卡线速约 {LINE:.0f} GB/s")
    if CPU_MODE:
        print("（CPU 模式：只检查脚本逻辑，数字没有参考意义）")
    print("-" * 72)
    rows = [
        ("FAIL" if bad else "OK  ", "数据校验（机内 all-reduce、跨机两个方向 send/recv）",
         "；".join(f"{e}（{k} 个 rank）" for e, k in sorted(bad.items())) if bad else "全部正确"),
        (grade(min(ar), 20, 8), "机内 8 卡 all-reduce 总线带宽",
         f"节点0 {ar[0]:.1f} / 节点1 {ar[1]:.1f} GB/s"),
        (grade(pair_bw, 0.6 * LINE, 0.25 * LINE), "跨机单对 send/recv（512 MiB）",
         f"{pair_bw:.1f} GB/s"),
        (grade(all8_bw, 0.7 * LINE, 0.3 * LINE), "跨机 8 对并发（PP 边界形态）",
         f"{all8_bw:.1f} GB/s 合计"),
        (grade(one_way_us, 100, 500, higher_is_better=False), "跨机 1 MiB 单程时延",
         f"{one_way_us:.0f} µs"),
    ]
    for g, name, val in rows:
        print(f"[{g}] {name}：{val}")

    H, src = hidden_size()
    per_tok = H * 2 * 2  # BF16 hidden + residual
    chunk = int(os.environ.get("CHUNKED_PREFILL", "16384"))
    batch = int(os.environ.get("MAX_RUNNING", "32"))
    print("-" * 72)
    print(f"按实测带宽估算 PP 边界传输（hidden={H}，{src}；每 token {per_tok // 1024} KiB）")
    for label, ntok in ((f"prefill 一块 {chunk} token", chunk), (f"decode 一步 {batch} 个请求", batch)):
        best = ntok * per_tok / (all8_bw * 1e9) * 1e3
        worst = best * G
        print(f"  {label}：约 {best:.2f} ms（8 个 TP rank 分摊）~ {worst:.2f} ms（每个 rank 各发一份）")
    print("  prefill 一块的计算本身是几百毫秒量级；decode 一步是几十毫秒量级。")
    print("=" * 72)
    failed = any(g.startswith("FAIL") for g, _, _ in rows)
    if any(e.startswith("机内") for e in bad):
        print("机内数据不对：运行 ./run_gpu_test.sh 看卡间 P2P；P2P 拷贝的数据不对就在 cluster.env 设 GPU_P2P=0")
    elif bad:
        print("只有跨机数据不对：日志里有 GDRDMA 时，多半是 GPUDirect RDMA 写错了显存，"
              "两台都加 NCCL_NET_GDR_LEVEL=LOC 关掉它再测；没有 GDRDMA 就查网卡和交换机")
    if failed:
        print("有 FAIL 项：先看本节点日志里 NCCL 是否走了 NET/IB、是否有 GDRDMA（run_nccl_test.sh 末尾会分析）")
    return failed


def main():
    if CPU_MODE:
        dev, dtype = torch.device("cpu"), torch.float32
        dist.init_process_group("gloo", timeout=TIMEOUT)
    else:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dev, dtype = torch.device("cuda", local), torch.bfloat16
        dist.init_process_group("nccl", timeout=TIMEOUT)

    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2 * G:
        if rank == 0:
            print(f"需要 2×{G} 个进程，实际 {world}")
        sys.exit(1)
    node = rank // G
    peer = (rank + G) % world
    groups = [dist.new_group(list(range(n * G, (n + 1) * G)), timeout=TIMEOUT) for n in range(2)]
    tp = groups[node]
    esize = torch.tensor([], dtype=dtype).element_size()

    def mk(nbytes):
        n = max(1, nbytes // SCALE // esize)
        return torch.ones(n, dtype=dtype, device=dev), n * esize

    res = {"rank": rank, "host": socket.gethostname()}
    if rank == 0:
        print(f"开始测试；通信超过 {int(TIMEOUT.total_seconds())} 秒没完成会报错退出（P2P 不通时 NCCL 常常是一直等）", flush=True)

    # 0. 数据校验
    res["errs"] = verify(rank, node, peer, tp, dev)

    # 1. 机内 all-reduce
    x, nb = mk(256 * MiB)
    dist.barrier()
    t = timed(lambda: dist.all_reduce(x, group=tp), iters=10)
    res["ar_busbw"] = nb * 2 * (G - 1) / G / t / 1e9
    del x

    # 2 / 3. 跨机 send/recv：senders 里的 rank 发给 rank+8
    def p2p(senders, buf, iters):
        def step():
            if rank in senders:
                dist.send(buf, dst=peer)
            elif peer in senders:
                dist.recv(buf, src=peer)
        dist.barrier()
        return timed(step, iters)

    x, nb = mk(512 * MiB)
    res["pair_t"], res["pair_nb"] = p2p({0}, x, iters=10), nb
    del x
    x, nb = mk(256 * MiB)
    res["all8_t"], res["all8_nb"] = p2p(set(range(G)), x, iters=10), nb
    del x

    # 4. 往返时延
    x, _ = mk(1 * MiB)

    def pingpong():
        if rank == 0:
            dist.send(x, dst=G)
            dist.recv(x, src=G)
        elif rank == G:
            dist.recv(x, src=0)
            dist.send(x, dst=0)

    dist.barrier()
    res["rtt"] = timed(pingpong, iters=50, warmup=5)

    allres = [None] * world
    dist.all_gather_object(allres, res)
    # 结论广播给所有进程：有 FAIL 时两台的 torchrun 都返回非 0，deploy.sh 据此停下
    verdict = [report(allres) if rank == 0 else None]
    dist.broadcast_object_list(verdict, src=0)
    if verdict[0] and rank == G:
        print("跨机测试有 FAIL 项，结果表在节点0的输出里")
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if verdict[0] else 0)


if __name__ == "__main__":
    main()
