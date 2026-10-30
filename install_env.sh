#!/usr/bin/env bash
# =====================================================================
# 裸机：创建 SGLang 的 Python 虚拟环境（两台机器各执行一次）
#
#   第一台：./install_env.sh
#           → 安装 SGLang，并把所有包的确切版本写进 requirements.lock
#   把整个目录（含 requirements.lock）同步到另一台，再执行 ./install_env.sh
#           → 严格按 lock 文件安装，两台版本完全一致
#
# 需要 uv：pip install uv，或 curl -LsSf https://astral.sh/uv/install.sh | sh
# 需要额外的包索引（如特定 CUDA 版本的 torch）时，设置 UV_EXTRA_INDEX_URL 后再运行。
# 想装指定的包（如本地 wheel），可以设置 SGLANG_SPEC，例如 SGLANG_SPEC=./sglang-xxx.whl
# =====================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SKIP_VENV_CHECK=1
# shellcheck source=common.sh
source "$HERE/common.sh"

[[ -n ${PY_ENV:-} ]] || die "cluster.env 里 PY_ENV 为空（用容器部署不需要这一步）"
command -v uv >/dev/null || die "找不到 uv：pip install uv，或 curl -LsSf https://astral.sh/uv/install.sh | sh"
LOCK="$HERE/requirements.lock"

if [[ -x $PY_ENV/bin/python ]]; then
  info "虚拟环境已存在：$PY_ENV"
else
  info "创建虚拟环境 $PY_ENV（Python $PY_VERSION）"
  uv venv --python "$PY_VERSION" "$PY_ENV"
fi
PY="$PY_ENV/bin/python"

if [[ -f $LOCK ]]; then
  info "按 $LOCK 安装，保证与另一台一致"
  uv pip sync --python "$PY" "$LOCK"
else
  spec="${SGLANG_SPEC:-sglang${SGLANG_VERSION:+==$SGLANG_VERSION}}"
  info "安装 $spec"
  uv pip install --python "$PY" --prerelease=allow "$spec"
  uv pip freeze --python "$PY" >"$LOCK"
  ok "已生成 $LOCK —— 把整个目录同步到另一台，再在那台运行本脚本"
fi

set +u
# shellcheck disable=SC1091
source "$PY_ENV/bin/activate"
set -u
"$PY" - <<'PY'
import importlib
for m in ("sglang", "torch", "flashinfer", "transformers"):
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
    except Exception as e:
        v = f"导入失败（{type(e).__name__}）"
    print(f"    {m:13s} {v}")
try:
    import torch
    print(f"    torch 编译所用 CUDA：{torch.version.cuda}")
except Exception:
    pass
PY
info "环境指纹：$(env_fingerprint)"
info "两台机器的环境指纹必须一致；preflight.sh 也会打印这个值"
if command -v nvcc >/dev/null; then
  info "nvcc：$(nvcc --version | grep -oE 'release [0-9.]+' | head -1)"
else
  warn "找不到 nvcc。flashinfer 在 SM120 上首次运行可能需要 JIT 编译：安装与上面 torch 的 CUDA 主版本一致的 CUDA Toolkit，并在 cluster.env 设置 CUDA_HOME"
fi
