#!/usr/bin/env bash
# =====================================================================
# 上线前自检。两台机器各跑一次：./preflight.sh
# 只读检查，不改系统。FAIL 必须先解决，WARN 视情况处理。
# =====================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

NFAIL=0
NWARN=0
bad()     { fail "$*"; NFAIL=$((NFAIL + 1)); }
meh()     { warn "$*"; NWARN=$((NWARN + 1)); }
section() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
have()    { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------
section "GPU"
GPU_MEM_MIB=0
if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,name,compute_cap,memory.total \
                                 --format=csv,noheader,nounits)
  if [[ ${#GPUS[@]} -eq 8 ]]; then ok "GPU 数量 8"; else bad "GPU 数量 ${#GPUS[@]}（需要 8）"; fi
  for g in "${GPUS[@]}"; do
    IFS=',' read -r idx name cap mem <<<"$g"
    name="$(xargs <<<"$name")"; cap="$(xargs <<<"$cap")"; mem="$(xargs <<<"$mem")"
    line="GPU$idx $name  SM ${cap/./}  ${mem} MiB"
    if [[ $cap != 12.0 ]]; then
      meh "$line —— 不是 SM120，这套脚本里的 SM120 绕行设置可能不需要"
    elif [[ $name != *"RTX PRO 6000"* ]]; then
      meh "$line —— 型号不是 RTX PRO 6000"
    elif ((mem < 90000)); then
      meh "$line —— 显存不足 96GB（如 6000D 的 84GB），KV 预算会明显变小"
    else
      ok "$line"
    fi
    if ((GPU_MEM_MIB == 0 || mem < GPU_MEM_MIB)); then GPU_MEM_MIB=$mem; fi
  done
  info "驱动版本：$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  busy="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  if ((busy > 0)); then meh "有 $busy 个进程正在占用 GPU（nvidia-smi 查看），启动前请停掉"; fi
  if nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | grep -qv Enabled; then
    meh "部分 GPU 没开 persistence 模式（裸机运行 host_setup.sh 处理）"
  fi
  if grep -q "Open Kernel Module" /proc/driver/nvidia/version 2>/dev/null; then
    OPEN_KM=1
    ok "使用开源内核模块"
  fi
else
  bad "nvidia-smi 不可用或看不到 GPU（容器需要 --gpus all）"
fi
if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
  meh "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES，启动前请 unset"
fi

# ---------------------------------------------------------------------
section "RDMA（${IB_HCA} 端口 ${IB_PORT:-1}）"
P="$(ib_sys)"
LL="$(rdma_link_layer)"
if [[ ! -d $P ]]; then
  devs=""
  for d in /sys/class/infiniband/*; do [[ -e $d ]] && devs+="$(basename "$d") "; done
  bad "找不到 $P。本机 RDMA 设备：${devs:-无}（容器需要 --device /dev/infiniband）"
else
  state="$(cat "$P/state" 2>/dev/null)"
  phys="$(cat "$P/phys_state" 2>/dev/null)"
  rate="$(cat "$P/rate" 2>/dev/null)"
  if [[ $state == *ACTIVE* ]]; then ok "端口状态 $state"; else bad "端口状态 ${state:-未知}（需要 ACTIVE）"; fi
  if [[ $phys == *LinkUp* ]]; then ok "物理链路 $phys"; else bad "物理链路 ${phys:-未知}（需要 LinkUp）"; fi
  rate_num="$(grep -oE '^[0-9]+' <<<"$rate" || echo 0)"
  if ((rate_num >= 400)); then ok "速率 $rate"; else meh "速率 ${rate:-未知}（预期 400 Gb/sec）"; fi
  info "链路层：$LL"

  if [[ $LL == Ethernet ]]; then
    want="${IB_GID_INDEX:-3}"
    v2=()
    for t in "$P"/gid_attrs/types/*; do
      i="$(basename "$t")"
      typ="$(cat "$t" 2>/dev/null)" || continue
      gid="$(cat "$P/gids/$i" 2>/dev/null)" || continue
      [[ $typ == *"RoCE v2"* ]] || continue
      [[ $gid == 0000:0000:0000:0000:0000:0000:0000:0000 ]] && continue
      nd="$(cat "$P/gid_attrs/ndevs/$i" 2>/dev/null)"
      mark=""
      [[ $gid == 0000:0000:0000:0000:0000:ffff:* ]] && mark="  ← IPv4，一般选这个"
      info "  RoCE v2 GID index $i  $gid  ${nd}${mark}"
      v2+=("$i")
    done
    if ((${#v2[@]} == 0)); then
      bad "没有可用的 RoCE v2 GID（网卡没配 IP？）"
    elif [[ " ${v2[*]} " == *" $want "* ]]; then
      ok "NCCL_IB_GID_INDEX=$want 是 RoCE v2"
    else
      bad "NCCL_IB_GID_INDEX=$want 不在上面的列表里，请把 cluster.env 的 IB_GID_INDEX 改成标了 IPv4 的那个"
    fi
  fi

  nds=""
  for d in "/sys/class/infiniband/$IB_HCA/device/net"/*; do [[ -e $d ]] && nds+="$(basename "$d") "; done
  info "该 RDMA 设备对应的网卡：${nds:-未知}"
  if [[ -n $nds && " $nds " != *" $NET_IFNAME "* ]]; then
    meh "NET_IFNAME=$NET_IFNAME 不是这块 RDMA 卡的网口（只用于建连，可以不同，但建议一致）"
  fi
fi
if [[ ! -d /dev/infiniband ]]; then
  bad "看不到 /dev/infiniband（容器需要 --device /dev/infiniband --cap-add IPC_LOCK）"
fi

# ---------------------------------------------------------------------
section "网络接口与连通性（$NET_IFNAME）"
if have ip && ip -4 -o addr show dev "$NET_IFNAME" >/dev/null 2>&1; then
  myip="$(ip -4 -o addr show dev "$NET_IFNAME" | awk '{print $4}' | cut -d/ -f1 | head -1)"
  if [[ -z $myip ]]; then
    bad "$NET_IFNAME 没有 IPv4 地址"
  else
    if [[ $myip == "$NODE0_ADDR" ]]; then
      ok "$NET_IFNAME = $myip，本机是节点0"
      peer="$NODE1_ADDR"
    elif [[ $myip == "$NODE1_ADDR" ]]; then
      ok "$NET_IFNAME = $myip，本机是节点1"
      peer="$NODE0_ADDR"
    else
      meh "$NET_IFNAME = $myip，既不是 NODE0_ADDR 也不是 NODE1_ADDR，检查 cluster.env"
      peer="$NODE0_ADDR"
    fi
    if have ping; then
      if ping -c 2 -W 1 -I "$NET_IFNAME" "$peer" >/dev/null 2>&1; then
        ok "能 ping 通对端（$peer）"
      else
        bad "ping 不通对端（$peer）"
      fi
    fi
  fi
  mtu="$(cat "/sys/class/net/$NET_IFNAME/mtu" 2>/dev/null)"
  if [[ $LL == Ethernet && -n $mtu ]] && ((mtu < 4200)); then
    meh "RoCE 网口 MTU=$mtu，建议 9000（交换机同步调整）"
  else
    info "MTU=${mtu:-未知}"
  fi
else
  bad "找不到网卡 $NET_IFNAME 或 ip 命令不可用"
fi

# ---------------------------------------------------------------------
section "驱动模块与系统限制"
kmaj="$(uname -r | cut -d. -f1)"; kmin="$(uname -r | cut -d. -f2)"
if [[ -d /sys/module/nvidia_peermem || -d /sys/module/nv_peer_mem ]]; then
  ok "nvidia-peermem 已加载（GPUDirect RDMA）"
elif [[ ${OPEN_KM:-0} == 1 ]] && ((kmaj > 5 || (kmaj == 5 && kmin >= 12))); then
  info "未加载 nvidia-peermem；开源内核模块 + 内核 $(uname -r) 下 NCCL 可以走 DMA-BUF，以 NCCL 测试里是否出现 GDRDMA 为准"
else
  meh "未加载 nvidia-peermem（宿主机执行 modprobe nvidia-peermem，或运行 host_setup.sh）"
fi
ml="$(ulimit -l)"
if [[ $ml == unlimited ]]; then
  ok "memlock = unlimited"
else
  bad "memlock = $ml，RDMA 注册内存会失败（容器加 --ulimit memlock=-1；裸机运行 host_setup.sh 后重新登录，或用 systemd 单元启动）"
fi
if [[ ! -f /.dockerenv ]]; then
  info "裸机部署：宿主机调优（IOMMU、ACS、NUMA、调频、防火墙）请运行 ./host_setup.sh 查看"
fi

# ---------------------------------------------------------------------
section "GPU 与网卡的 PCIe 距离"
if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
  topo="$(nvidia-smi topo -m 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g')"
  printf '%s\n' "$topo"
  nic="$(awk -v h="$IB_HCA" '$1 ~ /^NIC[0-9]+:$/ && $2 == h { sub(":", "", $1); print $1 }' <<<"$topo")"
  if [[ -n $nic ]]; then
    rel="$(awk -F'\t' -v n="$nic" '
      { k = $1; gsub(/ /, "", k) }
      k == n { for (i = 2; i <= 9; i++) { v = $i; gsub(/ /, "", v); printf "GPU%d:%s ", i - 2, v }; print "" }' <<<"$topo")"
    info "$IB_HCA（$nic）到各 GPU：$rel"
    near="$(grep -oE 'GPU[0-9]:(PIX|PXB)' <<<"$rel" | wc -l)"
    if ((near == 0)); then
      meh "网卡与所有 GPU 之间都要经过 CPU（NODE/SYS），NCCL 可能不启用 GPUDirect RDMA。PP 只需要几十 GB/s，通常仍够用，以 NCCL 测试结果为准"
    else
      ok "$near 张 GPU 与网卡在同一 PCIe 交换下（PIX/PXB），可走 GPUDirect RDMA"
    fi
  else
    meh "topo 图里没找到 $IB_HCA"
  fi
else
  info "跳过（没有 GPU）"
fi

# ---------------------------------------------------------------------
section "Python / SGLang 环境"
python3 - <<'PY'
import importlib
for m in ("torch", "sglang", "transformers", "flashinfer", "triton"):
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
    except Exception as e:
        v = f"未安装（{type(e).__name__}）"
    print(f"    {m:13s} {v}")
try:
    import torch
    if torch.cuda.is_available():
        nv = torch.cuda.nccl.version()
        nv = ".".join(map(str, nv)) if isinstance(nv, tuple) else str(nv)
        print(f"    CUDA {torch.version.cuda} · NCCL {nv} · capability {torch.cuda.get_device_capability(0)}")
    else:
        print("    torch 看不到 GPU")
except Exception as e:
    print(f"    torch 检查失败：{e}")
PY
info "Python：$(command -v python3)"
info "环境指纹：$(env_fingerprint) —— 两台机器必须一致"
if command -v nvcc >/dev/null; then
  info "nvcc：$(nvcc --version | grep -oE 'release [0-9.]+' | head -1)（应与上面 torch 的 CUDA 主版本一致）"
else
  meh "找不到 nvcc：flashinfer 在 SM120 上首次运行可能要 JIT 编译。装对应版本的 CUDA Toolkit，并在 cluster.env 设置 CUDA_HOME"
fi
if HELP="$(python3 -m sglang.launch_server --help 2>/dev/null)"; then
  for f in --pp-size --nnodes --dist-init-addr --kv-cache-dtype --attention-backend \
           --moe-runner-backend --fp8-gemm-backend --enable-pcie-oneshot-allreduce --speculative-algorithm; do
    if flag_in "$f" "$HELP"; then
      ok "支持 $f"
    else
      case $f in
        --pp-size | --nnodes | --dist-init-addr | --kv-cache-dtype) bad "不支持 $f，请升级 SGLang" ;;
        *) meh "不支持 $f（启动脚本会跳过或换成替代参数）" ;;
      esac
    fi
  done
else
  bad "python3 -m sglang.launch_server --help 运行失败"
fi

# ---------------------------------------------------------------------
section "模型（$MODEL_PATH）"
if [[ -f $MODEL_PATH/config.json ]]; then
  ok "config.json 存在"
  while IFS= read -r line; do
    case $line in
      FAIL:*) bad "${line#FAIL:}" ;;
      WARN:*) meh "${line#WARN:}" ;;
      OK:*)   ok "${line#OK:}" ;;
      *)      info "$line" ;;
    esac
  done < <(
    MODEL_PATH="$MODEL_PATH" GPU_MEM_MIB="$GPU_MEM_MIB" MEM_FRACTION="$MEM_FRACTION" \
    PP_LAYER_PARTITION="${PP_LAYER_PARTITION:-}" python3 - <<'PY'
import glob, json, os
mp = os.environ["MODEL_PATH"]
cfg = json.load(open(os.path.join(mp, "config.json")))
tc = cfg.get("text_config", cfg)
arch = ",".join(cfg.get("architectures") or ["?"])
L = int(tc.get("num_hidden_layers") or 0)
H = int(tc.get("hidden_size") or 0)
kv_dim = int(tc.get("kv_lora_rank") or 0) + int(tc.get("qk_rope_head_dim") or 0)
q = cfg.get("quantization_config") or tc.get("quantization_config") or {}
qm = q.get("quant_method") or q.get("quant_algo") or "无"
print(f"架构 {arch} · {L} 层 · hidden {H} · MLA KV 维度 {kv_dim} · 量化 {qm} · MTP 层 {tc.get('num_nextn_predict_layers', '?')}")
print(("OK:" if "GlmMoeDsa" in arch else "WARN:") + f"架构 {arch}")
print(("OK:" if str(qm).lower().startswith("fp8") else "WARN:") + f"量化方式 {qm}（本方案按 FP8 写）")

idx_path = os.path.join(mp, "model.safetensors.index.json")
files = sorted(glob.glob(os.path.join(mp, "*.safetensors")))
if os.path.exists(idx_path):
    need = set(json.load(open(idx_path))["weight_map"].values())
    missing = [f for f in need if not os.path.exists(os.path.join(mp, f))]
    if missing:
        print(f"FAIL:缺 {len(missing)} 个权重分片，例如 {missing[0]}（下载未完成？）")
    else:
        print(f"OK:权重分片齐全（{len(need)} 个）")
else:
    print("WARN:没有 model.safetensors.index.json")
total = sum(os.path.getsize(f) for f in files)
print(f"权重总大小 {total / 2**30:.0f} GiB")

if L and kv_dim and total:
    part = os.environ.get("PP_LAYER_PARTITION", "").strip()
    stages = [int(x) for x in part.split(",")] if part else [L // 2, L - L // 2]
    if sum(stages) != L:
        print(f"FAIL:PP_LAYER_PARTITION={part} 加起来不等于 {L}")
    mem_mib = float(os.environ.get("GPU_MEM_MIB") or 0) or 97887.0  # 96GB 卡的典型值
    frac = float(os.environ.get("MEM_FRACTION") or 0.85)
    w_gpu = total / 16
    budget = mem_mib * 2**20 * frac - w_gpu
    kv_tok = kv_dim * 2 * max(stages)  # BF16，TP 下每卡都存完整的 latent
    print(f"层划分 {stages[0]}/{stages[1]} · 每卡权重约 {w_gpu / 2**30:.0f} GiB · "
          f"KV 预算约 {budget / 2**30:.0f} GiB · 每 token KV {kv_tok / 1024:.1f} KiB")
    if budget <= 0:
        print("FAIL:按当前 MEM_FRACTION 放不下权重")
    else:
        print(f"粗估 KV 池约 {budget / kv_tok / 1e4:.0f} 万 token（所有请求共享；"
              f"以启动日志的 max_total_num_tokens 为准，激活和 CUDA Graph 还会再占一些）")
if H:
    print(f"PP 边界每 token 约 {H * 2 * 2 / 1024:.0f} KiB（BF16 hidden + residual）")
PY
  )
  if have df; then info "磁盘：$(df -h "$MODEL_PATH" | awk 'NR==2{print $1" 总 "$2"，可用 "$4}')"; fi
else
  bad "找不到 $MODEL_PATH/config.json（两台机器要放在同一路径）"
fi

# ---------------------------------------------------------------------
section "主机"
if have free; then info "内存：$(free -g | awk '/^Mem:/{print $2" GB 总量，"$7" GB 可用"}')"; fi
if have lscpu; then info "$(lscpu | grep -E '^NUMA node\(s\)' | xargs)"; fi
info "CPU 核数：$(nproc)"

# ---------------------------------------------------------------------
section "结果"
if ((NFAIL)); then
  fail "$NFAIL 项 FAIL，$NWARN 项 WARN —— 先解决 FAIL"
  exit 1
elif ((NWARN)); then
  warn "0 项 FAIL，$NWARN 项 WARN"
else
  ok "全部通过"
fi
