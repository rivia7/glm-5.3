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
# 只有节点0对外提供 HTTP（端口 HTTP_PORT）：ROUTER=dynamo 时是 Dynamo 前端（按会话固定 attention 组），
# 节点0 的 SGLang 跑在 Dynamo worker（dynamo_worker.py）里；ROUTER=sglang 时是 SGLang 自带的 HTTP 服务。
# 节点1 两种情况下都跑原生 SGLang。
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
# 层划分不对时要等权重加载完、处理第一个请求才报错，所以启动前先查
split="$(pp_split_check)"
case $split in
  OK:*) info "层划分：${split#OK:}" ;;
  WARN:*) warn "层划分：${split#WARN:}" ;;
  *)
    if [[ $DRY_RUN == 1 ]]; then warn "层划分：${split#FAIL:}"; else die "层划分：${split#FAIL:}"; fi
    ;;
esac

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
  HELP="--pp-size --fp8-gemm-backend --enable-pcie-oneshot-allreduce --cuda-graph-max-bs-decode --attn-dp-size --model-loader-extra-config --enable-metrics --watchdog-timeout --enable-hierarchical-cache --kv-events-config"
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
  # 新版 SGLang 把 --cuda-graph-max-bs 拆成了按阶段的参数
  if has_flag --cuda-graph-max-bs-decode; then
    GRAPH_ARGS=(--cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS")
  elif has_flag --cuda-graph-max-bs; then
    GRAPH_ARGS=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
  else
    GRAPH_ARGS=()
    warn "当前 SGLang 没有 --cuda-graph-max-bs(-decode)，CUDA_GRAPH_MAX_BS 不生效，用 SGLang 的默认值"
  fi
fi

ATTN_MODE="${ATTN_MODE:-dsa}"
KV_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
ATTN_DP="${ATTN_DP_SIZE:-1}"
case $ATTN_MODE in
  dense) ATTN_ARGS=(--attention-backend flashinfer) ;;  # 绕开 DSA，按稠密 MLA 计算
  dsa)   ATTN_ARGS=() ;;                                  # 交给 SGLang 自动选择 DSA 后端
  *)     die "ATTN_MODE 只能是 dense 或 dsa，收到: $ATTN_MODE" ;;
esac
case $KV_DTYPE in
  fp8_e4m3 | bf16) ;;
  *) die "KV_CACHE_DTYPE 只能是 fp8_e4m3 或 bf16，收到: $KV_DTYPE" ;;
esac
if [[ $ATTN_MODE == dense && $KV_DTYPE == fp8_e4m3 ]]; then
  # SGLang 0.5.23 只给 DSA 准备了 SM120 的 FP8 KV kernel；稠密注意力配 FP8 KV 在 SM120 上会输出乱码
  die "ATTN_MODE=dense 只能配 KV_CACHE_DTYPE=bf16；要用 FP8 KV 请设 ATTN_MODE=dsa"
fi
case $ATTN_DP in
  1 | 2 | 4 | 8) ;;
  *) die "ATTN_DP_SIZE 只能是 1、2、4、8（要能整除 TP8），收到: $ATTN_DP" ;;
esac
if ((ATTN_DP > 1)); then
  # attention 分成 ATTN_DP 组，每组 8/ATTN_DP 张卡、各自处理不同的请求；MoE 仍按 TP8 切
  if has_flag --attn-dp-size; then
    ATTN_ARGS+=(--attn-dp-size "$ATTN_DP")
  elif has_flag --enable-dp-attention; then
    ATTN_ARGS+=(--dp-size "$ATTN_DP" --enable-dp-attention)
    warn "当前 SGLang 没有 --attn-dp-size，改用旧写法 --dp-size $ATTN_DP --enable-dp-attention"
  else
    die "当前 SGLang 不支持 DP attention，把 ATTN_DP_SIZE 设成 1"
  fi
  if [[ $MTP == 1 ]]; then
    die "ENABLE_MTP=1 不能和 ATTN_DP_SIZE>1 同时用（SGLang 的 PP 投机解码不支持 DP attention）"
  fi
fi

# 入口：ROUTER=dynamo 时节点0 用 Dynamo 前端对外服务，按会话把请求固定到同一个 attention 组
ROUTER="${ROUTER:-sglang}"
case $ROUTER in
  dynamo | sglang) ;;
  *) die "ROUTER 只能是 dynamo 或 sglang，收到: $ROUTER" ;;
esac
SESSION_TTL="${SESSION_TTL:-1800}"
[[ $SESSION_TTL =~ ^[1-9][0-9]*$ ]] || die "SESSION_TTL 要是正整数（秒），收到: $SESSION_TTL"
WORKER_SYSTEM_PORT=$((HTTP_PORT + 1))  # Dynamo worker 的 /health 和 /metrics（含 sglang: 开头的 SGLang 指标）
WORKER_PORT=$((HTTP_PORT + 2))         # 只是传给节点0 SGLang 的 --port，Dynamo 模式下 SGLang 不起 HTTP 服务
KV_EVENTS_PORT=$((HTTP_PORT + 100))    # 各 attention 组的 KV 事件端口从这里起（第 i 组 +i），只绑本机回环地址
if [[ $ROUTER == dynamo ]]; then
  has_flag --kv-events-config || die "当前 SGLang 不支持 --kv-events-config，ROUTER=dynamo 用不了，改成 ROUTER=sglang"
  if [[ $NODE_RANK == 0 ]]; then
    [[ -z ${API_KEY:-} ]] || die "ROUTER=dynamo 不支持 API_KEY（Dynamo 前端不做鉴权）：要鉴权就在前面另加网关，或改用 ROUTER=sglang"
    DYN_PY="${DYNAMO_PY_ENV:-/opt/dynamo-venv}"
    DYN_PY="${DYN_PY%/}"
    DYN_PY="${DYN_PY%/bin/activate}/bin/python"
    if [[ ! -x $DYN_PY && $DRY_RUN != 1 ]]; then
      die "找不到 $DYN_PY：ROUTER=dynamo 要先在节点0 运行 ./install_dynamo.sh（或把 ROUTER 改成 sglang）"
    fi
    if ((ATTN_DP == 1)); then
      warn "ATTN_DP_SIZE=1 只有一组，会话固定没有作用；只是多一层 Dynamo 转发"
    fi
  fi
fi

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
  # 注意力和 KV（见 cluster.env 的 ATTN_MODE / KV_CACHE_DTYPE / ATTN_DP_SIZE）；SM120 上 MoE 走 Triton
  "${ATTN_ARGS[@]}"
  --kv-cache-dtype "$KV_DTYPE"
  --moe-runner-backend triton
  # 容量与调度
  --mem-fraction-static "$MEM_FRACTION"
  --context-length "$CTX"
  --max-running-requests "$RUNNING"
  --chunked-prefill-size "$CHUNKED_PREFILL"
  "${GRAPH_ARGS[@]}"
  # 思考内容解析器（显式写，兼容不认识 auto 的旧版本）
  --reasoning-parser glm45
)
if [[ $ROUTER == dynamo ]]; then
  # 两台传同样的参数：只有第一级流水（节点0）的调度进程会发布，每组一个端口；
  # Dynamo 靠这些事件知道哪一组缓存了哪些前缀。事件里带 token，所以只绑本机回环地址
  ARGS+=(--kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://127.0.0.1:${KV_EVENTS_PORT}\",\"bind\":true}"
         --port "$WORKER_PORT")
  # 工具调用由 Dynamo 前端解析（--dyn-tool-call-parser，见下面），SGLang 这边不能再设 --tool-call-parser
else
  ARGS+=(--tool-call-parser glm47 --host 0.0.0.0 --port "$HTTP_PORT")
fi

# ---------------- 可选参数（按版本支持情况添加）----------------
if has_flag --fp8-gemm-backend; then
  ARGS+=(--fp8-gemm-backend triton)   # SM120 上的 blockwise FP8 GEMM
else
  warn "当前 SGLang 没有 --fp8-gemm-backend；若加载 FP8 权重时报 kernel 不支持，需要升级或换 SM120 镜像"
fi
if [[ ${GPU_P2P:-1} == 0 ]]; then
  # 自定义 all-reduce（2 卡一组时才用）、PCIe-IPC all-reduce、旧版的 PCIe oneshot all-reduce 都靠 P2P，
  # SGLang 默认都不先验证 P2P 的数据对不对，P2P 坏了会悄悄算错
  ARGS+=(--disable-custom-all-reduce)
  export SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=0
  info "GPU_P2P=0：--disable-custom-all-reduce、关掉 PCIe-IPC all-reduce，机内 all-reduce 走 NCCL（主机共享内存）"
elif has_flag --enable-pcie-oneshot-allreduce; then
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

# ---------------- 主机内存二级前缀缓存（HiCache）----------------
HICACHE="${HICACHE_RATIO:-0}"
[[ $HICACHE =~ ^[0-9]+([.][0-9]+)?$ ]] || die "HICACHE_RATIO 要是非负数（0 = 关），收到: $HICACHE"
[[ $MODE == bringup ]] && HICACHE=0
if [[ ! $HICACHE =~ ^0+([.]0+)?$ ]]; then
  # 显存放不下的前缀 KV 写到主机内存，会话回访时拷回、不用重算；每张卡各开「显存 KV 池 × HICACHE_RATIO」的锁页内存
  has_flag --enable-hierarchical-cache || die "当前 SGLang 不支持 --enable-hierarchical-cache，把 HICACHE_RATIO 设成 0"
  ARGS+=(--enable-hierarchical-cache --hicache-ratio "$HICACHE")
else
  HICACHE=0
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

# ---------------- Dynamo（ROUTER=dynamo，只在节点0）----------------
USE_DYNAMO=0
if [[ $ROUTER == dynamo && $NODE_RANK == 0 ]]; then
  USE_DYNAMO=1
  # 前端和 worker 都在节点0：服务发现用本机目录，请求、响应和事件都走本机回环地址
  export DYN_DISCOVERY_BACKEND=file DYN_FILE_KV="$HERE/logs/dynamo_discovery"
  export DYN_REQUEST_PLANE=tcp DYN_TCP_RPC_HOST=127.0.0.1 DYN_TCP_RESPONSE_STREAM_HOST=127.0.0.1 DYN_EVENT_PLANE_HOST=127.0.0.1
  export DYN_SYSTEM_PORT="$WORKER_SYSTEM_PORT" DYN_LOG="${DYN_LOG:-info}"
  FRONT_ARGS=(
    --http-host 0.0.0.0
    --http-port "$HTTP_PORT"
    --router-mode kv                                   # 新会话按各组的前缀缓存命中和负载选组
    --router-session-affinity-ttl-secs "$SESSION_TTL"  # 带会话头的请求固定在该会话第一次选中的组
    --enable-anthropic-api                             # 和 SGLang 一样提供 /v1/messages
  )
  # GLM-5.3 的工具调用和思考内容由前端解析；清前缀缓存用 POST <节点0>:$WORKER_SYSTEM_PORT/engine/flush_cache
  ARGS+=(--dyn-tool-call-parser glm47 --dyn-reasoning-parser glm45 --engine-route flush_cache)
  # 前端跨 NUMA 节点运行会明显变慢（Dynamo 启动时会告警）：多 NUMA 的机器上把它绑到网卡所在的 NUMA 节点
  FRONT_PIN=()
  numa="$(cat "/sys/class/infiniband/${IB_HCA}/device/numa_node" 2>/dev/null || echo 0)"
  [[ $numa =~ ^[0-9]+$ ]] || numa=0
  cpus="$(cat "/sys/devices/system/node/node${numa}/cpulist" 2>/dev/null || true)"
  if [[ -n $cpus && -e /sys/devices/system/node/node1 ]] && command -v taskset >/dev/null; then
    FRONT_PIN=(taskset -c "$cpus")
    info "Dynamo 前端绑到 NUMA 节点 ${numa}（CPU ${cpus}）"
  fi
fi

# ---------------- 打印并启动 ----------------
mkdir -p "$HERE/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$HERE/logs/glm53_node${NODE_RANK}_${MODE}_${STAMP}.log"

info "节点 ${NODE_RANK} / 模式 ${MODE} / 注意力 ${ATTN_MODE}（${ATTN_DP} 组）/ KV ${KV_DTYPE} / MTP ${MTP} / HiCache ${HICACHE} / 入口 ${ROUTER}"
if ((USE_DYNAMO)); then
  info "本机负责第一级流水 + Dynamo 前端：http://${NODE0_ADDR}:${HTTP_PORT}（会话空闲 ${SESSION_TTL}s 后解除固定）"
  info "SGLang 的 /metrics 在 http://${NODE0_ADDR}:${WORKER_SYSTEM_PORT}/metrics，前端自己的在 HTTP_PORT 上"
elif [[ $NODE_RANK == 0 ]]; then
  info "本机负责第一级流水 + HTTP 服务：http://${NODE0_ADDR}:${HTTP_PORT}"
else
  info "本机负责第二级流水，不提供 HTTP；服务在 http://${NODE0_ADDR}:${HTTP_PORT}"
fi
info "NCCL 环境："
print_nccl_env

print_cmd() {  # print_cmd <命令> <参数...>：每个 -- 参数起一行，--api-key 的值打码
  local line="    $1" a mask_next=0
  shift
  for a in "$@"; do
    if ((mask_next)); then a="***"; mask_next=0; fi
    [[ $a == --api-key ]] && mask_next=1
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
}
info "启动命令："
if ((USE_DYNAMO)); then
  print_cmd "${FRONT_PIN[*]}${FRONT_PIN[*]:+ }$DYN_PY -m dynamo.frontend" "${FRONT_ARGS[@]}"
  print_cmd "$DYN_PY $HERE/dynamo_worker.py" "${ARGS[@]}"
else
  print_cmd "python3 -m sglang.launch_server" "${ARGS[@]}"
fi

if [[ $DRY_RUN == 1 ]]; then
  info "DRY_RUN=1，不启动"
  exit 0
fi
info "日志：$LOG"
if ((USE_DYNAMO)); then
  FLOG="$HERE/logs/dynamo_frontend_${MODE}_${STAMP}.log"
  info "Dynamo 前端日志：$FLOG"
  rm -rf "$DYN_FILE_KV"  # 上次留下的注册信息会让前端去连已经退出的 worker
  "${FRONT_PIN[@]}" "$DYN_PY" -m dynamo.frontend "${FRONT_ARGS[@]}" >"$FLOG" 2>&1 &
  FE_PID=$!
  # 前端一退出服务就断了：这时像按 Ctrl-C 一样停掉整个进程组（worker 和调用本脚本的 deploy.sh）
  (
    while kill -0 "$FE_PID" 2>/dev/null; do sleep 5; done
    warn "Dynamo 前端退出了（日志 $FLOG），停止服务"
    kill -INT 0
  ) &
  WATCH_PID=$!
  trap 'kill "$WATCH_PID" "$FE_PID" 2>/dev/null || true' EXIT
  "$DYN_PY" "$HERE/dynamo_worker.py" "${ARGS[@]}" 2>&1 | tee "$LOG"
else
  python3 -m sglang.launch_server "${ARGS[@]}" 2>&1 | tee "$LOG"
fi
