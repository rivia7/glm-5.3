#!/usr/bin/env bash
# =====================================================================
# 一键部署：配置检查 → 宿主机检查 → Python 环境 → 自检 → 跨机测试 → 启动 →（节点0）冒烟测试
# 任何一步失败就停下。两台用相同的选项各运行一次（先后无所谓，间隔不要太久）：
#   节点0：./deploy.sh 0 [bringup|serve] [选项]
#   节点1：./deploy.sh 1 [bringup|serve] [选项]
# 第一次用 bringup，跑通后再用 serve（默认）。
# 开始前先检查 cluster.env：必填项填了没有，网卡、RDMA 设备、GID、线速、CUDA 和本机对不对得上，
# 不对就给出建议值；有 FAIL 就停下。
#
# 选项：
#   --py-env PATH     SGLang 所在的 Python 环境，写虚拟环境目录或其 bin/activate，
#                     如 /home/test/sglang_env/bin/activate；覆盖 cluster.env 的 PY_ENV
#   --container       容器部署：在宿主机上创建或复用容器 glm53（镜像取 CONTAINER_IMAGE），
#                     后续步骤在容器里执行。此时 --py-env 指容器内的路径，不给就用镜像自带的环境
#   --host-apply      先执行 host_setup.sh --apply 修改宿主机设置（非 root 时用 sudo）
#   --gpu-test        加跑本机 GPU 测试：单卡带宽、卡与卡之间的带宽和时延、8 卡 all-reduce
#   --rdma-test       加跑不经过 NCCL 的纯 RDMA 带宽测试
#   --skip-nccl-test  跳过跨机 NCCL 测试
#   --no-smoke        节点0不自动跑冒烟测试
#   --dry-run         检查和测试照做，只打印启动命令，不启动
#
# 服务在前台运行，Ctrl-C 停止。两台是一个整体：一台退出，另一台也会退出，重启时两台一起重新运行。
# 要在断开 ssh 后继续运行，请放在 tmux 或 screen 里。
# =====================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG_ARGS=("$@")

usage() {
  echo "用法: $0 <0|1> [bringup|serve] [--py-env PATH] [--container] [--host-apply]" >&2
  echo "       [--gpu-test] [--rdma-test] [--skip-nccl-test] [--no-smoke] [--dry-run]（说明见脚本开头）" >&2
  exit 1
}

NODE_RANK="${1:-}"
[[ $NODE_RANK == 0 || $NODE_RANK == 1 ]] || usage
shift
STAGE=serve
PY_ENV_ARG="" HAVE_PY_ENV_ARG=0 CONTAINER=0 HOST_APPLY=0 GPU_TEST=0 RDMA_TEST=0 NCCL_TEST=1 SMOKE=1 DRY=0
while (($#)); do
  case $1 in
    bringup | serve) STAGE="$1" ;;
    --py-env)
      (($# >= 2)) || usage
      PY_ENV_ARG="$2" HAVE_PY_ENV_ARG=1
      shift
      ;;
    --container) CONTAINER=1 ;;
    --host-apply) HOST_APPLY=1 ;;
    --gpu-test) GPU_TEST=1 ;;
    --rdma-test) RDMA_TEST=1 ;;
    --skip-nccl-test) NCCL_TEST=0 ;;
    --no-smoke) SMOKE=0 ;;
    --dry-run) DRY=1 ;;
    *) usage ;;
  esac
  shift
done

IN_CONTAINER=0
if [[ -f /.dockerenv ]]; then IN_CONTAINER=1; fi
# 子脚本会各自 source common.sh，--py-env 通过环境变量传下去（见 common.sh）
if ((IN_CONTAINER)); then
  export GLM53_PY_ENV="$PY_ENV_ARG" GLM53_CUDA_HOME=""   # 容器里默认用镜像自带的 Python 和 CUDA
elif ((CONTAINER)); then
  export GLM53_PY_ENV=""                                 # 宿主机这一侧不需要 Python 环境
elif ((HAVE_PY_ENV_ARG)); then
  export GLM53_PY_ENV="$PY_ENV_ARG"
fi
_SKIP_VENV_CHECK=1
# shellcheck source=common.sh
source "$HERE/common.sh"

step() { printf '\n\033[1;35m########## %s ##########\033[0m\n' "$*"; }

# ---------------- 配置检查：必填项和与本机对应的项，填错了给出建议值 ----------------
NBAD=0 MY_RANK="" MY_IF=""
bad() { fail "$*"; NBAD=$((NBAD + 1)); }
is_ipv4() { [[ $1 =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; }

cuda_ok() {  # CUDA 版本是否 >= 12.8（SM120 的最低要求）
  local maj min
  IFS=. read -r maj min _ <<<"$1"
  ((${maj:-0} > 12 || (${maj:-0} == 12 && ${min:-0} >= 8)))
}

suggest_cuda_home() {  # /usr/local 下 >= 12.8 的 CUDA Toolkit，优先选主版本与 torch 一致的
  local d v match="" any=""
  for d in /usr/local/cuda-[0-9]*; do
    v="${d##*cuda-}"
    if [[ $v =~ ^[0-9]+\.[0-9]+$ && -x $d/bin/nvcc ]] && cuda_ok "$v"; then
      any="$d"
      if [[ ${v%%.*} == "$1" ]]; then match="$d"; fi
    fi
  done
  echo "${match:-$any}"
}

rdma_of_if() {  # 网卡对应的 RDMA 设备和端口，输出「设备 端口」，找不到时为空
  local d
  for d in /sys/class/net/"$1"/device/infiniband/*; do
    if [[ -e $d ]]; then
      echo "${d##*/} $(($(cat "/sys/class/net/$1/dev_port" 2>/dev/null || echo 0) + 1))"
      return
    fi
  done
}

roce_gid_index() {  # RoCE 端口 $1 上与 IPv4 地址 $2 对应的 RoCE v2 GID index，找不到时为空
  local a b c d want t
  IFS=. read -r a b c d <<<"$2"
  want="$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' "$a" "$b" "$c" "$d")"
  for t in "$1"/gid_attrs/types/*; do
    if [[ $(cat "$t" 2>/dev/null) == *"RoCE v2"* && $(cat "$1/gids/${t##*/}" 2>/dev/null) == "$want" ]]; then
      echo "${t##*/}"
      return
    fi
  done
}

check_required() {
  info "必填项（只有你知道的值，两台要一致）："
  if ! is_ipv4 "${NODE0_ADDR:-}" || ! is_ipv4 "${NODE1_ADDR:-}" || [[ $NODE0_ADDR == "$NODE1_ADDR" ]]; then
    bad "NODE0_ADDR / NODE1_ADDR 要填两台在 RDMA 网段上的不同 IPv4 地址（现在是 ${NODE0_ADDR:-空} / ${NODE1_ADDR:-空}）"
  else
    read -r MY_RANK MY_IF < <(ip -4 -o addr show 2>/dev/null | awk -v a="$NODE0_ADDR" -v b="$NODE1_ADDR" '
      { split($4, x, "/") } x[1] == a { print 0, $2; exit } x[1] == b { print 1, $2; exit }') || true
    if [[ -z $MY_RANK ]]; then
      bad "本机没有 NODE0_ADDR=$NODE0_ADDR 或 NODE1_ADDR=$NODE1_ADDR 这个 IP（ip -br -4 addr 查看本机 IP）"
    elif [[ $MY_RANK != "$NODE_RANK" ]]; then
      bad "本机 IP 是 NODE${MY_RANK}_ADDR（在 $MY_IF 上），却按节点$NODE_RANK 运行：节点编号写反了"
    else
      ok "NODE0_ADDR=$NODE0_ADDR  NODE1_ADDR=$NODE1_ADDR：本机是节点$NODE_RANK（IP 在 $MY_IF 上）"
    fi
  fi

  if [[ -f ${MODEL_PATH:-}/config.json ]]; then
    ok "MODEL_PATH=$MODEL_PATH"
  else
    bad "MODEL_PATH=${MODEL_PATH:-（空）} 下没有 config.json：两台都要把模型放在这个路径"
  fi

  if [[ -n ${API_KEY:-} ]]; then
    ok "API_KEY 已设置（不显示）"
  else
    warn "API_KEY 留空：服务不鉴权，能访问 $NODE0_ADDR:$HTTP_PORT 的人都能调用"
  fi

  local src=cluster.env
  if ((HAVE_PY_ENV_ARG)); then src=--py-env; fi
  if ((CONTAINER)); then
    info "PY_ENV：容器部署用镜像自带的环境${PY_ENV_ARG:+（--py-env 指定了容器内的 $PY_ENV_ARG）}"
  elif [[ -z $PY_ENV ]]; then
    warn "PY_ENV 留空：用 PATH 里的 python3（$(command -v python3 || echo 找不到)）"
  elif [[ -f $PY_ENV/bin/activate ]]; then
    ok "PY_ENV=$PY_ENV（来自 $src）"
  else
    info "PY_ENV=$PY_ENV（来自 $src）还不存在，后面会用 install_env.sh 创建"
  fi
}

check_net() {
  if [[ ! -e /sys/class/net/${NET_IFNAME:-?} ]]; then
    bad "找不到网卡 NET_IFNAME=${NET_IFNAME:-（空）}${MY_IF:+；本机 IP 在 $MY_IF 上，建议 NET_IFNAME=$MY_IF}"
  elif [[ -n $MY_IF && $MY_IF != "$NET_IFNAME" ]]; then
    warn "NET_IFNAME=$NET_IFNAME，但本机 IP 在 $MY_IF 上；一般应填 NET_IFNAME=$MY_IF"
  else
    ok "NET_IFNAME=$NET_IFNAME"
  fi
}

check_rdma() {
  local net="${MY_IF:-$NET_IFNAME}" my_ip="" hca port P ll rate num idx typ want
  if [[ $MY_RANK == 0 ]]; then my_ip="$NODE0_ADDR"; elif [[ $MY_RANK == 1 ]]; then my_ip="$NODE1_ADDR"; fi
  read -r hca port <<<"$(rdma_of_if "$net")"
  P="$(ib_sys)"
  if [[ ! -d $P ]]; then
    bad "找不到 RDMA 设备 IB_HCA=${IB_HCA:-（空）} 端口 ${IB_PORT:-1}${hca:+；$net 对应的是 $hca 端口 $port，建议 IB_HCA=$hca IB_PORT=$port}（rdma link 查看）"
    return
  fi
  ll="$(rdma_link_layer)"
  rate="$(cat "$P/rate" 2>/dev/null || true)"
  if [[ -n $hca && "$hca $port" != "$IB_HCA ${IB_PORT:-1}" ]]; then
    warn "IB_HCA=$IB_HCA IB_PORT=${IB_PORT:-1} 不是 $net 对应的 RDMA 设备；一般应填 IB_HCA=$hca IB_PORT=$port"
  else
    ok "IB_HCA=$IB_HCA IB_PORT=${IB_PORT:-1}：$ll，${rate:-速率未知}"
  fi

  num="$(grep -oE '^[0-9]+' <<<"$rate" || true)"
  if [[ -n $num && $num != "${LINE_RATE_GBPS:-}" ]]; then
    warn "LINE_RATE_GBPS=${LINE_RATE_GBPS:-（空）}，网卡当前速率是 $rate；网卡就是这个规格的话改成 LINE_RATE_GBPS=$num"
  elif [[ -n $num ]]; then
    ok "LINE_RATE_GBPS=$LINE_RATE_GBPS"
  fi

  if [[ $ll != Ethernet ]]; then
    info "IB_GID_INDEX：链路层是 $ll，一般留空"
    return
  fi
  idx="${IB_GID_INDEX:-3}"
  typ="$(cat "$P/gid_attrs/types/$idx" 2>/dev/null || true)"
  want=""
  if [[ -n $my_ip ]]; then want="$(roce_gid_index "$P" "$my_ip")"; fi
  if [[ $typ != *"RoCE v2"* ]]; then
    bad "IB_GID_INDEX=$idx 不是 RoCE v2 GID${want:+；本机 IP 对应的是 $want，建议 IB_GID_INDEX=$want}（preflight 会列出全部）"
  elif [[ -n $my_ip && -z $want ]]; then
    warn "没找到 $my_ip 对应的 RoCE v2 GID：这个 IP 不在 $IB_HCA 的网口上？"
  elif [[ -n $want && $want != "$idx" ]]; then
    warn "IB_GID_INDEX=$idx 不是本机 IP 对应的 GID；一般应填 IB_GID_INDEX=$want"
  else
    ok "IB_GID_INDEX=${IB_GID_INDEX:-（留空，用 3）}：RoCE v2${want:+，对应本机 IP}"
  fi
}

check_cuda() {  # 只在裸机上检查；common.sh 已把 CUDA_HOME/bin 放到 PATH 最前面
  local nvcc out ver torch_cuda sug
  nvcc="$(command -v nvcc || true)"
  torch_cuda="$(python3 -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
  sug="$(suggest_cuda_home "${torch_cuda%%.*}")"
  if [[ $sug == "${CUDA_HOME:-}" ]]; then sug=""; fi
  if [[ -n $CUDA_HOME && ! -x $CUDA_HOME/bin/nvcc ]]; then
    bad "CUDA_HOME=$CUDA_HOME 下没有 bin/nvcc${sug:+；建议 CUDA_HOME=$sug}"
    return
  fi
  if [[ -z $nvcc ]]; then
    warn "找不到 nvcc：flashinfer 在 SM120 上要 JIT 编译${sug:+；建议 CUDA_HOME=$sug}"
    return
  fi
  out="$("$nvcc" --version 2>/dev/null || true)"
  ver="$(grep -m1 -oE 'release [0-9]+\.[0-9]+' <<<"$out" || true)"
  ver="${ver#release }"
  if ! cuda_ok "$ver"; then
    bad "nvcc 版本是 ${ver:-未知}（$nvcc），SM120 至少要 CUDA 12.8${sug:+；建议 CUDA_HOME=$sug}"
  elif [[ -n $torch_cuda && ${ver%%.*} != "${torch_cuda%%.*}" ]]; then
    warn "nvcc $ver 与 torch 编译所用的 CUDA $torch_cuda 主版本不同${sug:+；建议 CUDA_HOME=$sug}"
  else
    ok "CUDA_HOME=${CUDA_HOME:-（留空）}：nvcc $ver${torch_cuda:+，torch 编译所用 CUDA $torch_cuda}"
  fi
}

check_config() {
  step "配置检查（cluster.env）"
  check_required
  info "和本机对应的项（两台的网卡名、设备名、路径要相同）："
  check_net
  check_rdma
  if ((CONTAINER == 0)); then check_cuda; fi
  if ((NBAD)); then
    die "cluster.env 有 $NBAD 项要改（见上面的 FAIL），改完记得同步到另一台"
  fi
  info "提醒：真实 IP 和 API_KEY 只写在节点上的 cluster.env 里，不要提交到公开仓库"
}

host_steps() {
  step "宿主机检查（host_setup.sh）"
  if ((HOST_APPLY == 0)); then
    "$HERE/host_setup.sh" || true
  elif [[ $EUID -eq 0 ]]; then
    "$HERE/host_setup.sh" --apply
  else
    sudo "$HERE/host_setup.sh" --apply
  fi
}

run_in_container() {
  command -v docker >/dev/null || die "找不到 docker"
  local name=glm53 image="${CONTAINER_IMAGE:-lmsysorg/sglang:latest}" models cur
  models="$(dirname "$MODEL_PATH")"
  step "容器 $name（$image）"
  if cur="$(docker inspect -f '{{.Config.Image}}' "$name" 2>/dev/null)"; then
    if [[ $cur != "$image" ]]; then
      die "已有容器 $name 用的镜像是 $cur，和 CONTAINER_IMAGE 不同；确认不再需要后 docker rm -f $name 再运行"
    fi
    if [[ $(docker inspect -f '{{.State.Running}}' "$name") == true ]]; then
      info "复用运行中的容器"
    else
      info "启动已有容器"
      docker start "$name" >/dev/null
    fi
  else
    info "创建容器（不设重启策略，不会开机自启）"
    docker run -d --name "$name" --gpus all --network host --ipc host --shm-size 32g \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      --device /dev/infiniband --cap-add IPC_LOCK \
      -v "$models:$models" -v "$HERE:/work" -w /work \
      "$image" sleep infinity >/dev/null
  fi
  local tty=()
  if [[ -t 0 && -t 1 ]]; then tty=(-it); fi
  # 在容器里重新执行本脚本：容器里有 /.dockerenv，会跳过宿主机检查和这一步
  exec docker exec "${tty[@]}" "$name" /work/deploy.sh "${ORIG_ARGS[@]}"
}

raise_limits() {
  # RDMA 注册内存需要 memlock 不限；软限制最多只能调到硬限制
  if ! ulimit -l unlimited 2>/dev/null; then
    warn "memlock 只能调到 $(ulimit -l)：先运行 sudo ./host_setup.sh --apply，再重新登录"
  fi
  ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
  ulimit -s 65536 2>/dev/null || true
}

env_steps() {
  step "Python 环境"
  if [[ -n $PY_ENV && ! -f $PY_ENV/bin/activate ]]; then
    if ((IN_CONTAINER)); then die "容器里没有 $PY_ENV/bin/activate"; fi
    if [[ $NODE_RANK == 1 && ! -f $HERE/requirements.lock ]]; then
      die "PY_ENV=$PY_ENV 不存在，也没有 requirements.lock：先在节点0运行本脚本创建环境，把整个目录同步过来再运行"
    fi
    info "PY_ENV=$PY_ENV 不存在，用 install_env.sh 创建"
    "$HERE/install_env.sh"
    # 两台要装同一份 lock，装完先停下，避免另一台在跨机测试里空等
    if [[ $NODE_RANK == 0 ]]; then
      ok "环境已创建。把整个目录（含 requirements.lock）同步到节点1，等节点1的环境也就绪后，两台再同时运行 deploy.sh"
    else
      ok "环境已按 requirements.lock 创建。确认两台的环境指纹一致后，两台再同时运行 deploy.sh"
    fi
    exit 0
  fi
  local ver
  if ! ver="$(python3 -c 'import sglang; print(sglang.__version__)' 2>/dev/null)"; then
    die "$(command -v python3) 里导入不了 sglang：用 --py-env 指定 SGLang 所在的环境，或在 cluster.env 里设置 PY_ENV"
  fi
  ok "Python $(command -v python3) · SGLang $ver"
}

test_steps() {
  step "上线前自检（preflight.sh）"
  "$HERE/preflight.sh" || die "自检有 FAIL 项，解决后再运行（另一台如果已经往下走了，Ctrl-C 停掉）"

  if ((GPU_TEST)); then
    step "本机 GPU 测试（run_gpu_test.sh）"
    "$HERE/run_gpu_test.sh" || die "本机 GPU 测试有 FAIL 项，看上面的报告（另一台如果已经往下走了，Ctrl-C 停掉）"
  fi

  if ((RDMA_TEST)); then
    step "纯 RDMA 带宽测试（rdma_perftest.sh）"
    if [[ $NODE_RANK == 0 ]]; then
      "$HERE/rdma_perftest.sh" server
    else
      local i
      for ((i = 1; ; i++)); do  # 节点0的 server 可能还没起来，连不上就过一会儿再试
        if "$HERE/rdma_perftest.sh" client; then break; fi
        ((i < 30)) || die "试了 30 次都连不上节点0的 ib_write_bw server"
        info "10 秒后重试（第 $i 次）"
        sleep 10
      done
    fi
  fi

  if ((NCCL_TEST)); then
    step "跨机 NCCL 测试（run_nccl_test.sh）"
    "$HERE/run_nccl_test.sh" "$NODE_RANK" || die "NCCL 测试没有通过，看上面的结果表和日志分析"
  fi
}

SMOKE_PID=""
stop_smoke() { if [[ -n $SMOKE_PID ]]; then kill "$SMOKE_PID" 2>/dev/null || true; fi; }

launch_steps() {
  step "启动（$STAGE 模式）"
  if ((DRY)); then
    DRY_RUN=1 "$HERE/launch_glm53_pp2.sh" "$NODE_RANK" "$STAGE"
    return
  fi
  if [[ $NODE_RANK == 0 ]] && ((SMOKE)); then
    local args=(--long 20000) log
    if [[ $STAGE == serve ]]; then args=(--long 60000 --bench 8); fi
    mkdir -p "$HERE/logs"
    log="$HERE/logs/smoke_${STAGE}_$(date +%Y%m%d_%H%M%S).log"
    info "服务就绪后自动跑冒烟测试（smoke_test.py ${args[*]}），结果也写入 $log"
    python3 "$HERE/smoke_test.py" --wait 7200 "${args[@]}" > >(tee "$log") 2>&1 &
    SMOKE_PID=$!
    trap stop_smoke EXIT
  fi
  "$HERE/launch_glm53_pp2.sh" "$NODE_RANK" "$STAGE"
}

if ((IN_CONTAINER)); then
  info "在容器里运行：配置检查和宿主机检查已在宿主机上做过，这里跳过；默认用镜像自带的 Python 环境"
else
  check_config
  host_steps
  if ((CONTAINER)); then run_in_container; fi  # exec 进容器，不再返回
fi
raise_limits
env_steps
test_steps
launch_steps
