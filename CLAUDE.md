# CLAUDE.md

GLM-5.3 FP8 · 两台 8×RTX PRO 6000 · PP2×TP8（SGLang）部署脚本，各文件作用见 README.md 的文件表（增删脚本时同步更新）。

- GitHub 上的仓库是公开的：提交前检查暂存区，不要把真实的 `API_KEY`、内网 IP 提交进 `cluster.env`。
- 推送 git：在 main 上 commit 后直接 push，不开分支、不走 PR。
- `cluster.env` 是唯一配置，同时被 `common.sh` 用 bash `source`、被 `smoke_test.py` 的 `load_env` 按 `#` 截断解析：只写 `KEY=字面值`，值里不能有 `#`，也不要用 `$VAR`、`$(...)`。
- shell 脚本统一 `source "$HERE/common.sh"`：它负责读配置、激活 `PY_ENV`（目录或 `bin/activate` 都行），并提供 `info/ok/warn/die`、`setup_nccl_env`、`flag_in` 等公共函数；不要自己再读 `cluster.env`。
- 子脚本各自 source `common.sh`，命令行要覆盖配置只能走环境变量：`deploy.sh` 导出 `GLM53_PY_ENV` / `GLM53_CUDA_HOME`，`common.sh` 用它们覆盖 `cluster.env`（设成空表示不用）。
- `deploy.sh` 是一键入口，按顺序调用其它脚本；容器模式下用 `docker exec` 在容器里重新执行自己，靠 `/.dockerenv` 判断已在容器内。
- 不要加开机自启（systemd 单元、`docker run --restart`）：服务由 `deploy.sh` 在前台运行，这是有意的设计。
- 两台各跑一份脚本、在跨机步骤会合：会中断流程的检查要让两台一起失败（如 `nccl_xnode_test.py` 把结论广播给所有 rank），否则一台停下，另一台会卡在会合处。
- `launch_glm53_pp2.sh` 增删 SGLang 参数时，同步 `preflight.sh` 的参数检查：无条件传入的缺失算 FAIL，经 `has_flag` 降级的算 WARN。
- 开着 `pipefail` 时，不要把退出码可能非 0 的命令直接接 `grep -q`（`ib_write_bw --help` 的退出码就是 1）：先存进变量再 grep。
- 需要 root 的检查（`ufw status`、`lspci -vvv` 等）在非 root 时用 `host_setup.sh` 的 `skip` 标成没检查，不能当作通过。
- 脚本要在两台 GPU 节点上运行（`smoke_test.py` 只用标准库，连得上服务就能跑），开发机上只做语法检查：
  - `for f in *.sh; do bash -n "$f"; done`
  - `python3 -c 'import ast,sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]' *.py`（`py_compile` 会生成 `__pycache__`）
- README、注释、终端提示和 commit message 都用中文。
