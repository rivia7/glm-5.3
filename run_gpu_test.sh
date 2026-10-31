#!/usr/bin/env bash
# =====================================================================
# 本机 GPU 测试：单卡带宽、卡与卡之间的带宽和时延（按 PIX/PXB/PHB/NODE/SYS 汇总）、8 卡 NCCL all-reduce。
# 只测本机，两台各跑一次，不需要同时运行，约 1 分钟：./run_gpu_test.sh
# 报告同时写入 logs/gpu_test_<时间>.log；有 FAIL 项时以非 0 退出。服务在跑时不要测。
# all-reduce 用与部署相同的 NCCL 设置；要对比别的设置，在前面加环境变量，如：
#   NCCL_P2P_LEVEL=NODE ./run_gpu_test.sh
# =====================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

setup_nccl_env
export MODEL_PATH CHUNKED_PREFILL MAX_RUNNING   # all-reduce 按 decode 一步、prefill 一块的消息大小测
mkdir -p "$HERE/logs"
LOG="$HERE/logs/gpu_test_$(date +%Y%m%d_%H%M%S).log"
info "报告：$LOG"

python3 "$HERE/gpu_local_test.py" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
