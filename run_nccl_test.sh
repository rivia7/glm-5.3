#!/usr/bin/env bash
# =====================================================================
# 跨机 NCCL 测试。两台机器分别执行（先后无所谓，一分钟内都启动即可）：
#   节点0：./run_nccl_test.sh 0
#   节点1：./run_nccl_test.sh 1
# 结果表在节点0输出；两台的日志末尾都会分析 NCCL 是否走了 RDMA / GPUDirect。
# 结果表里有 FAIL 项时，两台都以非 0 退出。
# =====================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

NODE_RANK="${1:-}"
[[ $NODE_RANK == 0 || $NODE_RANK == 1 ]] || die "用法: $0 <0|1>"
command -v torchrun >/dev/null || die "找不到 torchrun（需要在装了 PyTorch 的环境里运行）"

setup_nccl_env
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
export MODEL_PATH CHUNKED_PREFILL MAX_RUNNING LINE_RATE_GBPS
mkdir -p "$HERE/logs"
LOG="$HERE/logs/nccl_test_node${NODE_RANK}.log"

info "NCCL 环境："
print_nccl_env
info "日志：$LOG"

torchrun --nnodes 2 --nproc-per-node 8 --node-rank "$NODE_RANK" \
  --master-addr "$NODE0_ADDR" --master-port "$((DIST_PORT + 100))" \
  "$HERE/nccl_xnode_test.py" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

# ---------------- 日志分析 ----------------
printf '\n\033[1m== 本节点 NCCL 日志分析 ==\033[0m\n'
if grep -q "NET/IB : Using" "$LOG"; then
  ok "走的是 RDMA：$(grep -m1 -oE 'NET/IB : Using.*' "$LOG")"
elif grep -q "NET/Socket : Using" "$LOG"; then
  fail "NCCL 退回了 TCP Socket（带宽会差一个数量级）。检查 /dev/infiniband、memlock、IB_HCA 名称"
else
  warn "日志里没找到网络选择信息"
fi
n_net="$(grep -c 'via NET/IB' "$LOG" || true)"
n_gdr="$(grep -cE 'via NET/IB/[0-9]+(/[0-9]+)?/GDRDMA' "$LOG" || true)"
if ((n_net > 0)); then
  if ((n_gdr > 0)); then
    ok "跨机通道 $n_net 条，其中 $n_gdr 条用了 GPUDirect RDMA"
  else
    warn "跨机通道 $n_net 条，但都没用 GPUDirect RDMA（数据绕道主机内存）。检查 nvidia-peermem、PCIe ACS、IOMMU"
  fi
fi
if grep -q "NCCL WARN" "$LOG"; then
  warn "日志里有 NCCL WARN（前 5 条）："
  grep -m5 "NCCL WARN" "$LOG" | sed 's/^/    /'
fi
exit "$rc"
