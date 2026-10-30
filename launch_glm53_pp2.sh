#!/usr/bin/env bash
# =====================================================================
# GLM-5.3（FP8）· PP2 × TP8 · 两台 8×RTX PRO 6000（SM120）· 单路 400G RDMA
#
# 用法（两台都要启动，先后无所谓，间隔不要太久）：
#   节点0：./launch_glm53_pp2.sh 0 [bringup|serve]
#   节点1：./launch_glm53_pp2.sh 1 [bringup|serve]
#   只看最终命令、不启动：DRY_RUN=1 ./launch_glm53_pp2.sh 0 serve
#
# bringup：第一次启动用。关 CUDA Graph、32K 上下文、8 并发、不开 MTP、
#          NCCL 打 INFO 日志，先确认各个 kernel 和跨机通信都能跑通。
# serve  ：正式服务，参数取自 cluster.env。
# 只有节点0对外提供 HTTP（端口 HTTP_PORT）。
# =====================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

NODE_RANK="${1:-}"
MODE="${2:-serve}"
DRY_RUN="${DRY_RUN:-0}"
[[ $NODE_RANK == 0 || $NODE_RANK == 1 ]] || die "用法: $0 <0|1> [bringup|serve]"
[[ $MODE == bringup || $MODE == serve ]] || die "模式只能是 bringup 或 serve，收到: $MODE"
if [[ ! -f $MODEL_PATH/config.json && $DRY_RUN != 1 ]]; then
  die "找不到 $MODEL_PATH/config.json（两台机器要放在同一路径）"
fi

# ---------------- 网络 ----------------
setup_nccl_env
if [[ $MODE == bringup ]]; then
  export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
fi

# ---------------- SM120 必需 ----------------
# DeepGEMM 依赖 SM100 才有的 TMEM/tcgen05 指令，SM120 上必须关掉
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_ENABLE_DEEP_GEMM=0
export SGLANG_DISABLE_DEEP_GEMM=1
# expandable_segments 与 PCIe oneshot allreduce 冲突（会在 pcie_allreduce.cu 崩溃）
unset PYTORCH_CUDA_ALLOC_CONF
export SAFETENSORS_FAST_GPU=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
if [[ -n ${PP_LAYER_PARTITION:-} ]]; then
  export SGLANG_PP_LAYER_PARTITION="$PP_LAYER_PARTITION"
fi

# ---------------- 探测当前 SGLang 支持哪些参数 ----------------
if HELP="$(python3 -m sglang.launch_server --help 2>/dev/null)"; then
  :
elif [[ $DRY_RUN == 1 ]]; then
  warn "当前环境没有 SGLang，DRY_RUN 下假设所有可选参数都支持"
  HELP="--pp-size --fp8-gemm-backend --enable-pcie-oneshot-allreduce --model-loader-extra-config --enable-metrics --watchdog-timeout"
else
  die "python3 -m sglang.launch_server --help 运行失败，检查 SGLang 环境"
fi
has_flag() { flag_in "$1" "$HELP"; }
has_flag --pp-size || die "当前 SGLang 不支持 --pp-size，请升级"

# ---------------- 按模式取参数 ----------------
if [[ $MODE == bringup ]]; then
  CTX=32768
  RUNNING=8
  MTP=0
  GRAPH_ARGS=(--disable-cuda-graph)
else
  CTX="$CONTEXT_LENGTH"
  RUNNING="$MAX_RUNNING"
  MTP="${ENABLE_MTP:-0}"
  GRAPH_ARGS=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
fi

case "${ATTN_MODE:-dense}" in
  dense) ATTN_ARGS=(--attention-backend flashinfer) ;;  # 绕开 DSA，按稠密 MLA 计算
  dsa)   ATTN_ARGS=() ;;                                  # 交给 SGLang 自动选择 DSA 后端
  *)     die "ATTN_MODE 只能是 dense 或 dsa，收到: $ATTN_MODE" ;;
esac

ARGS=(
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --trust-remote-code
  # 并行：机内 TP8，跨机 PP2
  --tp-size 8
  --pp-size 2
  --nnodes 2
  --node-rank "$NODE_RANK"
  --dist-init-addr "${NODE0_ADDR}:${DIST_PORT}"
  # SM120：BF16 KV（FP8 KV 在 SM120 上会输出乱码）、MoE 走 Triton
  "${ATTN_ARGS[@]}"
  --kv-cache-dtype bf16
  --moe-runner-backend triton
  # 容量与调度
  --mem-fraction-static "$MEM_FRACTION"
  --context-length "$CTX"
  --max-running-requests "$RUNNING"
  --chunked-prefill-size "$CHUNKED_PREFILL"
  "${GRAPH_ARGS[@]}"
  # 解析器（显式写，兼容不认识 auto 的旧版本）
  --reasoning-parser glm45
  --tool-call-parser glm47
  --host 0.0.0.0
  --port "$HTTP_PORT"
)

# ---------------- 可选参数（按版本支持情况添加）----------------
if has_flag --fp8-gemm-backend; then
  ARGS+=(--fp8-gemm-backend triton)   # SM120 上的 blockwise FP8 GEMM
else
  warn "当前 SGLang 没有 --fp8-gemm-backend；若加载 FP8 权重时报 kernel 不支持，需要升级或换 SM120 镜像"
fi
if has_flag --enable-pcie-oneshot-allreduce; then
  ARGS+=(--enable-pcie-oneshot-allreduce)
else
  ARGS+=(--disable-custom-all-reduce)
  info "没有 --enable-pcie-oneshot-allreduce，改用 --disable-custom-all-reduce（机内 all-reduce 走 NCCL）"
fi
if has_flag --model-loader-extra-config; then
  ARGS+=(--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}')
fi
if has_flag --watchdog-timeout; then
  ARGS+=(--watchdog-timeout 1800)
fi
if has_flag --enable-metrics; then
  ARGS+=(--enable-metrics)
fi
if [[ -n ${API_KEY:-} ]]; then
  ARGS+=(--api-key "$API_KEY")
fi

if [[ $MTP == 1 ]]; then
  # 必须：否则 NEXTN 会退化成 EAGLE，把整个模型再加载一遍，直接 OOM
  export SGLANG_ENABLE_SPEC_V2=True
  ARGS+=(--speculative-algorithm NEXTN
         --speculative-num-steps 3
         --speculative-eagle-topk 1
         --speculative-num-draft-tokens 4)
  warn "MTP 在 PP 下没有公开验证；启动报错或日志里 accept length 很低就关掉"
fi

# ---------------- 打印并启动 ----------------
mkdir -p "$HERE/logs"
LOG="$HERE/logs/glm53_node${NODE_RANK}_${MODE}_$(date +%Y%m%d_%H%M%S).log"

info "节点 $NODE_RANK / 模式 $MODE / 注意力 ${ATTN_MODE:-dense} / MTP $MTP"
if [[ $NODE_RANK == 0 ]]; then
  info "本机负责第一级流水 + HTTP 服务：http://${NODE0_ADDR}:${HTTP_PORT}"
else
  info "本机负责第二级流水，不提供 HTTP；服务在 http://${NODE0_ADDR}:${HTTP_PORT}"
fi
info "NCCL 环境："
print_nccl_env
info "启动命令："
masked=() mask_next=0
for a in "${ARGS[@]}"; do
  if ((mask_next)); then masked+=("***"); mask_next=0; else masked+=("$a"); fi
  [[ $a == --api-key ]] && mask_next=1
done
line='    python3 -m sglang.launch_server'
for a in "${masked[@]}"; do
  if [[ $a == --* ]]; then
    printf '%s \\\n' "$line"
    line="      $a"
  elif [[ $a =~ ^[A-Za-z0-9_./:,=+-]+$ ]]; then
    line+=" $a"
  elif [[ $a != *"'"* ]]; then
    line+=" '$a'"
  else
    line+=" $(printf '%q' "$a")"
  fi
done
printf '%s\n' "$line"

if [[ $DRY_RUN == 1 ]]; then
  info "DRY_RUN=1，不启动"
  exit 0
fi
info "日志：$LOG"
python3 -m sglang.launch_server "${ARGS[@]}" 2>&1 | tee "$LOG"
