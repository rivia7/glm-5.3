#!/usr/bin/env bash
# =====================================================================
# 一键部署：宿主机检查 → Python 环境 → 自检 → 跨机测试 → 启动 →（节点0）冒烟测试
# 任何一步失败就停下。两台用相同的选项各运行一次（先后无所谓，间隔不要太久）：
#   节点0：./deploy.sh 0 [bringup|serve] [选项]
#   节点1：./deploy.sh 1 [bringup|serve] [选项]
# 第一次用 bringup，跑通后再用 serve（默认）。
#
# 选项：
#   --py-env PATH     SGLang 所在的 Python 环境，写虚拟环境目录或其 bin/activate，
#                     如 /home/test/sglang_env/bin/activate；覆盖 cluster.env 的 PY_ENV
#   --container       容器部署：在宿主机上创建或复用容器 glm53（镜像取 CONTAINER_IMAGE），
#                     后续步骤在容器里执行。此时 --py-env 指容器内的路径，不给就用镜像自带的环境
#   --host-apply      先执行 host_setup.sh --apply 修改宿主机设置（非 root 时用 sudo）
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
  echo "       [--rdma-test] [--skip-nccl-test] [--no-smoke] [--dry-run]（说明见脚本开头）" >&2
  exit 1
}

NODE_RANK="${1:-}"
[[ $NODE_RANK == 0 || $NODE_RANK == 1 ]] || usage
shift
STAGE=serve
PY_ENV_ARG="" HAVE_PY_ENV_ARG=0 CONTAINER=0 HOST_APPLY=0 RDMA_TEST=0 NCCL_TEST=1 SMOKE=1 DRY=0
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
  info "在容器里运行：跳过宿主机检查，默认用镜像自带的 Python 环境"
else
  host_steps
  if ((CONTAINER)); then run_in_container; fi  # exec 进容器，不再返回
fi
raise_limits
env_steps
test_steps
launch_steps
