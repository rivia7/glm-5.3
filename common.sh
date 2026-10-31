#!/usr/bin/env bash
# 公共函数，由其它脚本 source，不要直接执行。
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=cluster.env
source "$_COMMON_DIR/cluster.env"
# deploy.sh 通过环境变量覆盖裸机设置：GLM53_PY_ENV 来自 --py-env；在容器里两项都置空，用镜像自带的 Python 和 CUDA
if [[ -n ${GLM53_PY_ENV+x} ]]; then PY_ENV="$GLM53_PY_ENV"; fi
if [[ -n ${GLM53_CUDA_HOME+x} ]]; then CUDA_HOME="$GLM53_CUDA_HOME"; fi
# PY_ENV 可以写虚拟环境目录，也可以写它的 bin/activate
PY_ENV="${PY_ENV:-}"
PY_ENV="${PY_ENV%/}"
PY_ENV="${PY_ENV%/bin/activate}"

info() { printf '\033[36m[INFO]\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m[ OK ]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[WARN]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$*" >&2; }
die()  { fail "$*"; exit 1; }

# ---- 裸机：CUDA Toolkit 和 Python 虚拟环境 ----
if [[ -n ${CUDA_HOME:-} ]]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
fi
if [[ -n ${PY_ENV:-} ]]; then
  if [[ -f $PY_ENV/bin/activate ]]; then
    case $- in *u*) _had_u=1; set +u ;; *) _had_u=0 ;; esac
    # shellcheck disable=SC1091
    source "$PY_ENV/bin/activate"
    if [[ $_had_u == 1 ]]; then set -u; fi
  elif [[ ${_SKIP_VENV_CHECK:-0} != 1 ]]; then
    warn "PY_ENV=$PY_ENV 不存在：裸机先运行 ./install_env.sh，或用 deploy.sh --py-env 指定已有环境；用容器请把 PY_ENV 留空"
  fi
fi

# 判断某个参数是否出现在 sglang --help 的输出里：flag_in --pp-size "$HELP"
flag_in() { grep -qE -- "(^|[^a-z-])$1([^a-z-]|$)" <<<"$2"; }

ib_sys() { echo "/sys/class/infiniband/${IB_HCA}/ports/${IB_PORT:-1}"; }

rdma_link_layer() {
  local f
  f="$(ib_sys)/link_layer"
  if [[ -r $f ]]; then cat "$f"; else echo unknown; fi
}

setup_nccl_env() {
  # 跨机必须走 RDMA。单机教程里常见的 NCCL_IB_DISABLE=1 千万不要照抄。
  unset NCCL_IB_DISABLE
  # 强制用 IB/RoCE：网络初始化失败时直接报错，而不是悄悄退回 TCP（能跑但慢很多）
  export NCCL_NET="${NCCL_NET:-IB}"
  # 只用这一块网卡；前导 = 表示精确匹配（避免 mlx5_1 匹配到 mlx5_10）
  export NCCL_IB_HCA="=${IB_HCA}:${IB_PORT:-1}"
  export NCCL_SOCKET_IFNAME="$NET_IFNAME"
  export GLOO_SOCKET_IFNAME="$NET_IFNAME"
  if [[ $(rdma_link_layer) == Ethernet ]]; then
    export NCCL_IB_GID_INDEX="${IB_GID_INDEX:-3}"   # RoCE v2
  elif [[ -n ${IB_GID_INDEX:-} ]]; then
    export NCCL_IB_GID_INDEX="$IB_GID_INDEX"
  fi
  if [[ ${GPU_P2P:-1} == 0 ]]; then
    # 卡间 P2P 不可用（拷贝的数据不对或会卡住）：机内改走主机共享内存
    export NCCL_P2P_DISABLE=1
    unset NCCL_P2P_LEVEL
  else
    # 机内没有 NVLink，走 PCIe P2P（与社区 GLM-5 on RTX PRO 6000 配方一致）
    export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
  fi
  export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"
  export NCCL_ALLOC_P2P_NET_LL_BUFFERS="${NCCL_ALLOC_P2P_NET_LL_BUFFERS:-1}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
}

print_nccl_env() { env | grep -E '^(NCCL_|GLOO_)' | sort | sed 's/^/    /'; }

# 本机在 NET_IFNAME 上的 IP，以及另一台机器的 IP（判断不出时为空）
local_ip() { ip -4 -o addr show dev "$NET_IFNAME" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1; }
peer_ip() {
  local me
  me="$(local_ip)"
  if [[ $me == "$NODE0_ADDR" ]]; then echo "$NODE1_ADDR"
  elif [[ $me == "$NODE1_ADDR" ]]; then echo "$NODE0_ADDR"
  fi
}

# 检查 PP 两级的层划分（PP_LAYER_PARTITION 留空时同 SGLang 默认：均分，多出的层给第二级）。
# 输出一行：OK:说明、WARN:说明 或 FAIL:说明。
# GLM-5.3 的 DSA indexer 只在 indexer_types 为 full 的层算 top-k，shared 层复用上一层的 top-k。
# 第二级从 shared 层开始时，SGLang 要从第一级把 top-k 传过去；稠密注意力下第一级不算 top-k，第一个请求就会报错
pp_split_check() {
  MODEL_PATH="$MODEL_PATH" PP_LAYER_PARTITION="${PP_LAYER_PARTITION:-}" ATTN_MODE="${ATTN_MODE:-dense}" python3 - <<'PY'
import json, os
try:
    with open(os.path.join(os.environ["MODEL_PATH"], "config.json")) as f:
        cfg = json.load(f)
except Exception as e:
    print(f"FAIL:读不到 {os.environ['MODEL_PATH']}/config.json（{type(e).__name__}），没法检查层划分")
    raise SystemExit
tc = cfg.get("text_config", cfg)
L = int(tc.get("num_hidden_layers") or 0)
types = tc.get("indexer_types") or []
shared = lambda i: i < len(types) and types[i] == "shared"
# 离均分最近的两个合法起点，作为建议值
near = sorted(sorted((i for i in range(1, L) if not shared(i)), key=lambda i: abs(i - L / 2))[:2])
good = " 或 ".join(f"{i},{L - i}" for i in near)
part = os.environ["PP_LAYER_PARTITION"].strip()
src = f"PP_LAYER_PARTITION={part}" if part else "PP_LAYER_PARTITION 留空（按 SGLang 默认均分）"
try:
    stages = [int(x) for x in part.split(",")] if part else [L // 2, L - L // 2]
except ValueError:
    stages = []
if len(stages) != 2 or min(stages) < 1 or sum(stages) != L:
    print(f"FAIL:{src}：要写成两个正整数，加起来等于模型的 {L} 层，如 {good}")
    raise SystemExit
b = stages[0]
desc = f"第一级 0–{b - 1} 层（{b} 层），第二级 {b}–{L - 1} 层（{L - b} 层）"
if shared(b):
    fix = f"改成 PP_LAYER_PARTITION={good}"
    if os.environ["ATTN_MODE"] == "dense":
        print(f"FAIL:{src}，{desc}。第 {b} 层的 DSA indexer 是 shared（复用上一层的 top-k），"
              f"稠密注意力下第二级从这层开始，处理第一个请求时就会报错；第二级要从 full 层开始，{fix}")
    else:
        print(f"WARN:{src}，{desc}。第 {b} 层的 DSA indexer 是 shared，"
              f"SGLang 要把 top-k 随 hidden states 一起跨机传（没有验证过）；{fix} 可以省掉")
else:
    print(f"OK:{src}，{desc}")
PY
}

# Python 环境指纹：所有已安装包及版本的哈希。两台机器输出一致才算环境相同
env_fingerprint() {
  python3 - <<'PY'
import hashlib
from importlib import metadata
pkgs = sorted({f"{d.metadata['Name'].lower()}=={d.version}" for d in metadata.distributions() if d.metadata["Name"]})
print(f"{hashlib.sha256(chr(10).join(pkgs).encode()).hexdigest()[:12]}（{len(pkgs)} 个包）")
PY
}
