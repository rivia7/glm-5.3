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
  # 机内没有 NVLink，走 PCIe P2P（与社区 GLM-5 on RTX PRO 6000 配方一致）
  export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
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

# Python 环境指纹：所有已安装包及版本的哈希。两台机器输出一致才算环境相同
env_fingerprint() {
  python3 - <<'PY'
import hashlib
from importlib import metadata
pkgs = sorted({f"{d.metadata['Name'].lower()}=={d.version}" for d in metadata.distributions() if d.metadata["Name"]})
print(f"{hashlib.sha256(chr(10).join(pkgs).encode()).hexdigest()[:12]}（{len(pkgs)} 个包）")
PY
}
