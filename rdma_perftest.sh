#!/usr/bin/env bash
# =====================================================================
# 裸 RDMA 带宽测试（不经过 NCCL），先确认网络本身没问题。需要 perftest（ib_write_bw）：
#   Ubuntu/Debian: apt install perftest    RHEL 系: dnf install perftest    或 DOCA-OFED 自带
#
#   节点0：./rdma_perftest.sh server [GPU 序号]
#   节点1：./rdma_perftest.sh client [GPU 序号]
#
# 不带 GPU 序号：测主机内存之间的 RDMA，看网卡和交换机本身能不能跑满 400G。
# 带 GPU 序号  ：测显存之间的 RDMA（GPUDirect），需要 perftest 编译时带 CUDA。
#                选 preflight 里显示 PIX/PXB 的那张卡（离网卡最近）。
# =====================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SKIP_VENV_CHECK=1
# shellcheck source=common.sh
source "$HERE/common.sh"

ROLE="${1:-}"
GPU="${2:-}"
[[ $ROLE == server || $ROLE == client ]] || die "用法: $0 server|client [GPU 序号]"
command -v ib_write_bw >/dev/null || die "找不到 ib_write_bw，请安装 perftest"

ARGS=(-d "$IB_HCA" -i "${IB_PORT:-1}" -F --report_gbits -q 4 -s 1048576 -D 10 -p 18515)
if [[ $(rdma_link_layer) == Ethernet ]]; then
  ARGS+=(-x "${IB_GID_INDEX:-3}")
fi
if [[ -n $GPU ]]; then
  # --help 的退出码不是 0，开着 pipefail 时不能直接接 grep
  help_text="$(ib_write_bw --help 2>&1 || true)"
  if grep -q -- '--use_cuda' <<<"$help_text"; then
    ARGS+=("--use_cuda=$GPU")
  else
    die "这个 perftest 没有 CUDA 支持，无法测 GPUDirect；可以只测主机内存，GPUDirect 交给 run_nccl_test.sh 判断"
  fi
fi

info "ib_write_bw ${ARGS[*]}"
if [[ $ROLE == server ]]; then
  info "等待节点1连接（节点1执行 ./rdma_perftest.sh client ${GPU}）"
  ib_write_bw "${ARGS[@]}"
else
  ib_write_bw "${ARGS[@]}" "$NODE0_ADDR"
fi
info "参考：${LINE_RATE_GBPS:-400}G 网卡的 BW average 应在 $(( ${LINE_RATE_GBPS:-400} * 9 / 10 )) Gb/s 左右或更高"
