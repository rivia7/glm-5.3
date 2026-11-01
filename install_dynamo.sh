#!/usr/bin/env bash
# =====================================================================
# 节点0：安装 NVIDIA Dynamo（ROUTER=dynamo 时用；节点1 跑原生 SGLang，不用装）
#
#   ./install_dynamo.sh        （deploy.sh 在节点0 发现没装时会自动运行）
#
# 在 DYNAMO_PY_ENV 建一个叠加在 PY_ENV 上的虚拟环境：Dynamo 自己的包装在这里，SGLang、torch 等
# 直接用 PY_ENV 里的（.pth 把 PY_ENV 的 site-packages 加进来），PY_ENV 本身不改，两台的环境指纹不变。
# ai-dynamo 声明的依赖里有它自己配的 SGLang 版本，还有一大批只有模拟、规划功能才用到的包（jax 等），
# 所以用 --no-deps，只装 Dynamo 和 worker、前端启动时要导入的两个包（redis、blake3）。
# 卸载：删掉 DYNAMO_PY_ENV 目录。下载慢时设置 UV_INDEX_URL（如 https://mirrors.aliyun.com/pypi/simple/）再运行。
# =====================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

DYN_ENV="${DYNAMO_PY_ENV:-}"
DYN_ENV="${DYN_ENV%/}"
DYN_ENV="${DYN_ENV%/bin/activate}"
VER="${DYNAMO_VERSION:-}"
[[ -n $DYN_ENV ]] || die "cluster.env 里 DYNAMO_PY_ENV 为空"
[[ -n $VER ]] || die "cluster.env 里 DYNAMO_VERSION 为空"
[[ $DYN_ENV != "${PY_ENV:-}" ]] || die "DYNAMO_PY_ENV 不能和 PY_ENV 是同一个目录（Dynamo 要装在叠加的环境里，不改 PY_ENV）"
UV="$(command -v uv || true)"
if [[ -z $UV && -x $HOME/.local/bin/uv ]]; then UV="$HOME/.local/bin/uv"; fi
[[ -n $UV ]] || die "找不到 uv：pip install uv，或 curl -LsSf https://astral.sh/uv/install.sh | sh"

# PY_ENV（common.sh 已激活；为空时是 PATH 里的 python3）：Dynamo 要用同一个解释器和其中的 SGLang
read -r BASE_PY SGL_VER < <(python3 -c 'import sys, sglang; print(sys._base_executable, sglang.__version__)' 2>/dev/null) ||
  die "$(command -v python3) 里导入不了 sglang：先装好 PY_ENV（install_env.sh）"
mapfile -t BASE_SITES < <(python3 -c 'import site; print("\n".join(site.getsitepackages()))')
info "SGLang $SGL_VER（$(command -v python3)）· 解释器 $BASE_PY"

if [[ -x $DYN_ENV/bin/python ]]; then
  info "虚拟环境已存在：$DYN_ENV"
else
  info "创建虚拟环境 $DYN_ENV"
  "$UV" venv --python "$BASE_PY" "$DYN_ENV"
fi
DYN_PY="$DYN_ENV/bin/python"
SP="$("$DYN_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
{
  echo "# install_dynamo.sh 生成：把 PY_ENV 的包加进来，Dynamo 用其中的 SGLang"
  for d in "${BASE_SITES[@]}"; do echo "import site; site.addsitedir('$d')"; done
} >"$SP/zz_glm53_base_env.pth"

info "安装 ai-dynamo==$VER（--no-deps）"
"$UV" pip install --python "$DYN_PY" --no-deps \
  "ai-dynamo==$VER" "ai-dynamo-runtime==$VER" "redis>=6.2,<9" "blake3>=1,<2"

# 两个入口都要导入得了，SGLang 必须是 PY_ENV 里那份
if ! out="$("$DYN_PY" - "$SGL_VER" 2>&1 <<'PY'
import sys
import sglang
import dynamo.frontend.main  # noqa: F401
import dynamo.sglang.main  # noqa: F401
from importlib.metadata import version
assert sglang.__version__ == sys.argv[1], f"SGLang 版本是 {sglang.__version__}，和 PY_ENV 的 {sys.argv[1]} 不同"
print(f"ai-dynamo {version('ai-dynamo')} · SGLang {sglang.__version__}（{sglang.__file__}）")
PY
)"; then
  printf '%s\n' "$out" | tail -20 >&2
  die "Dynamo 装好了但导入失败（见上面的报错）；缺包时用 $UV pip install --python $DYN_PY --no-deps <包名> 补上"
fi
ok "$(tail -1 <<<"$out")"
info "ROUTER=dynamo 时 launch_glm53_pp2.sh 会用 $DYN_PY 启动 Dynamo 前端和节点0 的 SGLang"
