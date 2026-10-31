#!/usr/bin/env bash
# =====================================================================
# 裸机宿主机准备（两台机器各执行一次）
#
#   ./host_setup.sh            只检查，列出会改什么（防火墙和 ACS 要用 sudo 运行才能检查）
#   sudo ./host_setup.sh --apply   实际修改
#
# 会改的项目（都可以安全重复执行）：
#   1. GPU persistence 模式（驱动常驻，避免每次初始化 GPU 的延迟和偶发失败）
#   2. 加载 nvidia-peermem 并设为开机加载（GPUDirect RDMA）
#   3. memlock 不限（RDMA 注册内存需要；对新登录的会话生效）
#   4. kernel.numa_balancing=0（避免内核迁移 GPU 进程的内存页）
#   5. CPU 调频策略设为 performance（仅本次开机有效）
#   6. 防火墙放行另一台机器的 IP（ufw / firewalld）
#
# 只检查、不会自动改的项目（需要进 BIOS 或改内核启动参数）：
#   IOMMU、PCIe ACS
# =====================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SKIP_VENV_CHECK=1
# shellcheck source=common.sh
source "$HERE/common.sh"

APPLY=0
[[ ${1:-} == --apply ]] && APPLY=1
if [[ $APPLY == 1 && $EUID -ne 0 ]]; then die "--apply 需要 root（sudo ./host_setup.sh --apply）"; fi
have()    { command -v "$1" >/dev/null 2>&1; }
section() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
NTODO=0
NSKIP=0
skip() { warn "$*"; NSKIP=$((NSKIP + 1)); }  # 权限或工具不够、没能检查的项
todo() {  # 需要修改的项：检查模式只打印，--apply 模式执行
  local desc="$1"; shift
  if [[ $APPLY == 1 ]]; then
    info "执行：$desc"
    "$@" || fail "失败：$desc"
  else
    warn "待修改：$desc"
    NTODO=$((NTODO + 1))
  fi
}
write_file() { printf '%s\n' "$2" >"$1"; }

# ---------------------------------------------------------------------
section "1. GPU persistence 模式"
if have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
  if nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | grep -qv Enabled; then
    if have systemctl && systemctl cat nvidia-persistenced.service >/dev/null 2>&1; then
      todo "启用 nvidia-persistenced 服务" systemctl enable --now nvidia-persistenced
    else
      todo "nvidia-smi -pm 1（重启后需重新执行）" nvidia-smi -pm 1
    fi
  else
    ok "所有 GPU 已开启 persistence 模式"
  fi
  if grep -q "Open Kernel Module" /proc/driver/nvidia/version 2>/dev/null; then
    ok "使用开源内核模块（Blackwell 必需）"
  else
    warn "看起来不是开源内核模块。Blackwell 只支持开源内核模块（nvidia-open），请确认驱动安装方式"
  fi
else
  fail "nvidia-smi 不可用"
fi

# ---------------------------------------------------------------------
section "2. nvidia-peermem（GPUDirect RDMA）"
if [[ -d /sys/module/nvidia_peermem ]]; then
  ok "已加载"
elif modinfo nvidia-peermem >/dev/null 2>&1; then
  todo "modprobe nvidia-peermem" modprobe nvidia-peermem
else
  warn "系统里没有 nvidia-peermem 模块。新内核 + 开源内核模块时 NCCL 可以改走 DMA-BUF，以 NCCL 测试里是否出现 GDRDMA 为准"
fi
if [[ -f /etc/modules-load.d/nvidia-peermem.conf ]]; then
  ok "已设为开机加载"
elif modinfo nvidia-peermem >/dev/null 2>&1; then
  todo "写 /etc/modules-load.d/nvidia-peermem.conf（开机加载）" \
    write_file /etc/modules-load.d/nvidia-peermem.conf nvidia-peermem
fi

# ---------------------------------------------------------------------
section "3. memlock"
LIMITS=/etc/security/limits.d/90-glm53-memlock.conf
if [[ -f $LIMITS ]]; then
  ok "$LIMITS 已存在"
else
  todo "写 $LIMITS（memlock 不限，重新登录后生效）" \
    write_file "$LIMITS" $'*    soft    memlock    unlimited\n*    hard    memlock    unlimited'
fi
info "当前会话 memlock = $(ulimit -l)"

# ---------------------------------------------------------------------
section "4. NUMA 自动均衡"
SYSCTL=/etc/sysctl.d/90-glm53.conf
nb="$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo ?)"
if [[ $nb == 0 ]]; then
  ok "kernel.numa_balancing = 0"
else
  todo "kernel.numa_balancing=0（写入 $SYSCTL 并立即生效）" \
    bash -c "printf 'kernel.numa_balancing = 0\n' >'$SYSCTL' && sysctl -q -p '$SYSCTL'"
fi

# ---------------------------------------------------------------------
section "5. CPU 调频"
gov_file=/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
if [[ -r $gov_file ]]; then
  gov="$(cat "$gov_file")"
  if [[ $gov == performance ]]; then
    ok "scaling_governor = performance"
  else
    # shellcheck disable=SC2016  # \$f 要在 bash -c 内部展开
    todo "所有 CPU 设为 performance（当前 $gov；仅本次开机有效，长期请用 tuned/cpupower）" \
      bash -c 'for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance >"$f"; done'
  fi
else
  info "没有 cpufreq 接口（BIOS 接管调频），跳过"
fi

# ---------------------------------------------------------------------
section "6. 防火墙"
PEER="$(peer_ip)"
if [[ -z $PEER ]]; then
  warn "$NET_IFNAME 上的 IP 既不是 NODE0_ADDR 也不是 NODE1_ADDR，无法确定对端，跳过"
elif [[ $EUID -ne 0 ]]; then
  # 非 root 时 ufw status 直接报错，会被误判成没有防火墙
  skip "查看防火墙规则需要 root，跳过"
elif have ufw && ufw status 2>/dev/null | grep -q "Status: active"; then
  if ufw status 2>/dev/null | grep -qwF "$PEER"; then
    ok "ufw 已有放行 $PEER 的规则"
  else
    todo "ufw allow from $PEER（放行对端全部端口）" ufw allow from "$PEER"
  fi
elif have firewall-cmd && firewall-cmd --state >/dev/null 2>&1; then
  if firewall-cmd --zone=trusted --list-sources 2>/dev/null | grep -qwF "$PEER"; then
    ok "firewalld 已把 $PEER 加入 trusted 区域"
  else
    todo "firewalld：把 $PEER 加入 trusted 区域" \
      bash -c "firewall-cmd -q --permanent --zone=trusted --add-source='$PEER' && firewall-cmd -q --reload"
  fi
else
  ok "没有检测到运行中的 ufw / firewalld（如有其它防火墙或安全组，请手动放行 $PEER）"
fi

# ---------------------------------------------------------------------
section "只检查：IOMMU 与 PCIe ACS（需要改 BIOS 或内核参数）"
if compgen -G "/sys/class/iommu/*" >/dev/null; then
  if grep -qwE 'iommu=pt' /proc/cmdline; then
    ok "IOMMU 已开启，且是 passthrough 模式（iommu=pt）"
  else
    warn "IOMMU 已开启但不是 passthrough 模式，可能拖慢或阻断 GPU P2P 与 GPUDirect RDMA。建议内核参数加 iommu=pt（或在 BIOS 关闭 IOMMU）"
  fi
else
  ok "IOMMU 未开启"
fi
if [[ $EUID -eq 0 ]] && have lspci; then
  # 只数会把 P2P 转到 CPU 的 ReqRedir / CmpltRedir / EgressCtrl；SrcValid 只校验请求来自哪个设备，开着不影响 P2P
  nacs="$(lspci -vvv 2>/dev/null | grep -cE 'ACSCtl:.*(ReqRedir|CmpltRedir|EgressCtrl)[+]' || true)"
  if ((nacs == 0)); then
    ok "没有 PCIe 桥开启 ACS 重定向（ReqRedir / CmpltRedir / EgressCtrl），P2P 不会被绕到 CPU"
  elif compgen -G "/sys/class/iommu/*" >/dev/null; then
    warn "$nacs 个 PCIe 桥开启了 ACS 重定向，GPU 之间、GPU 与网卡之间的 P2P 会被绕到 CPU。IOMMU 开着时内核启动会自己打开 ACS，BIOS 里关 ACS 不管用：在 BIOS 里关掉 IOMMU，或每次开机后用 setpci 关（见 README 0b 第 2 步）"
  else
    warn "$nacs 个 PCIe 桥开启了 ACS 重定向，GPU 之间、GPU 与网卡之间的 P2P 会被绕到 CPU。请在 BIOS 里关闭 ACS"
  fi
else
  skip "检查 ACS 需要 root 和 lspci，跳过"
fi
info "BIOS 里还应确认：Above 4G Decoding 与 Resizable BAR 已开启"

# ---------------------------------------------------------------------
section "结果"
if [[ $APPLY == 1 ]]; then
  ok "已执行完毕。memlock 需要重新登录才生效；IOMMU/ACS 需要手动处理"
elif ((NTODO)); then
  warn "$NTODO 项待修改，确认后执行：sudo ./host_setup.sh --apply"
elif ((NSKIP == 0)); then
  ok "无需修改"
fi
if ((NSKIP)); then
  warn "$NSKIP 项没有检查（见上面的 WARN），用 sudo ./host_setup.sh 再看一次"
fi
