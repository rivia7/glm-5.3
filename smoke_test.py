#!/usr/bin/env python3
"""GLM-5.3 服务冒烟测试（只用 Python 标准库，在任意能访问节点0的机器上运行）。

  python3 smoke_test.py                  # 模型列表 / 原样输出（查乱码）/ 思考分离 / 工具调用
  python3 smoke_test.py --long 32000     # 另加约 32K token 的长上下文找暗号
  python3 smoke_test.py --bench 8        # 另加 8 并发吞吐粗测
  python3 smoke_test.py --wait 3600      # 先等服务就绪再测（和启动脚本同时运行时用）

默认从同目录 cluster.env 读取地址、端口、API Key 和模型名，可用参数覆盖。
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "明月松间照，清泉石上流。"
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询某个城市当前的天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
    },
}]


def load_env(path):
    env = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return env


class Client:
    def __init__(self, base, key, model, timeout):
        self.base, self.key, self.model, self.timeout = base, key, model, timeout
        # 不走系统代理：服务在内网
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def req(self, method, path, body=None, timeout=None):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.base + path, data=data, method=method)
        r.add_header("Content-Type", "application/json")
        if self.key:
            r.add_header("Authorization", f"Bearer {self.key}")
        try:
            with self.opener.open(r, timeout=timeout or self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}") from None

    def chat(self, messages, max_tokens=2048, timeout=None, **extra):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"reasoning_effort": "low", "clear_thinking": True},
        }
        body.update(extra)
        t0 = time.perf_counter()
        r = self.req("POST", "/v1/chat/completions", body, timeout)
        return r, time.perf_counter() - t0


def t_models(c):
    ids = [m.get("id") for m in c.req("GET", "/v1/models").get("data", [])]
    return bool(ids), ("可用模型：" + ", ".join(ids)) if ids else "没有返回模型"


def t_echo(c):
    r, dt = c.chat([{"role": "user", "content": f"请把下面这句话原样输出，不要添加任何其他内容：{TARGET}"}])
    msg = r["choices"][0]["message"]
    content = msg.get("content") or ""
    toks = r.get("usage", {}).get("completion_tokens", 0)
    problems = []
    if TARGET.rstrip("。") not in content:
        problems.append(f"输出不符（可能是乱码）：{content[:80]!r}")
    if "</think>" in content:
        problems.append("content 里有 </think>，reasoning parser 没生效")
    note = (f"{toks} tokens / {dt:.1f}s ≈ {toks / dt:.1f} tok/s（含首 token）；"
            f"思考内容{'已分离到 reasoning_content' if msg.get('reasoning_content') else '为空'}")
    return not problems, "；".join(problems + [note]) if problems else f"原样输出正确；{note}"


def t_tool(c):
    r, dt = c.chat([{"role": "user", "content": "北京现在天气怎么样？请调用工具查询。"}], tools=TOOLS)
    ch = r["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    if ch.get("finish_reason") != "tool_calls" or not calls:
        return False, (f"没有解析出工具调用（finish_reason={ch.get('finish_reason')}，"
                       f"content={(ch['message'].get('content') or '')[:80]!r}），检查 --tool-call-parser glm47")
    fn = calls[0]["function"]
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        return False, f"参数不是合法 JSON：{fn.get('arguments')!r}"
    if fn.get("name") != "get_weather" or "city" not in args:
        return False, f"调用内容不对：{fn}"
    return True, f"get_weather({json.dumps(args, ensure_ascii=False)})，{dt:.1f}s"


def t_long(c, n_tokens):
    line = "Line {i}: the quick brown fox jumps over the lazy dog near the quiet river bank.\n"
    n = max(10, n_tokens // 18)  # 粗估每行约 18 个 token
    lines = [line.format(i=i) for i in range(n)]
    lines.insert(n // 2, "IMPORTANT: the secret code is 7391-BLUE-WHALE.\n")
    prompt = "".join(lines) + "\nWhat is the secret code mentioned above? Answer with the code only."
    r, dt = c.chat([{"role": "user", "content": prompt}], timeout=3600)
    content = (r["choices"][0]["message"].get("content") or "").strip()
    pt = r.get("usage", {}).get("prompt_tokens", "?")
    return "7391" in content, f"prompt {pt} tokens，耗时 {dt:.1f}s，回答 {content[:60]!r}"


def t_bench(c, conc):
    def one(_):
        r, _ = c.chat([{"role": "user", "content": "用三句话介绍一下香港的地理特点。"}], max_tokens=512)
        return r.get("usage", {}).get("completion_tokens", 0)

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(conc) as ex:
        toks = sum(ex.map(one, range(conc)))
    dt = time.perf_counter() - t0
    return toks > 0, f"{conc} 并发共 {toks} tokens / {dt:.1f}s ≈ {toks / dt:.1f} tok/s（总吞吐，含 prefill）"


def wait_ready(c, seconds):
    """轮询 /v1/models，返回了模型就返回 True，超时返回 False。

    Dynamo 前端比 SGLang 先起来，模型加载完、worker 注册之前，模型列表是空的。
    """
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if c.req("GET", "/v1/models", timeout=10).get("data"):
                return True
        except RuntimeError as e:  # HTTP 4xx（比如 API Key 不对）也说明服务已经起来了
            if not str(e).startswith("HTTP 5"):
                return True
        except OSError:  # 连接被拒或超时：还在加载
            pass
        time.sleep(10)
    return False


def main():
    env = load_env(os.path.join(HERE, "cluster.env"))
    ap = argparse.ArgumentParser(description="GLM-5.3 服务冒烟测试")
    ap.add_argument("--base", default=f"http://{env.get('NODE0_ADDR') or '127.0.0.1'}:{env.get('HTTP_PORT') or 30000}")
    ap.add_argument("--key", default=env.get("API_KEY", ""))
    ap.add_argument("--model", default=env.get("SERVED_NAME") or "glm-5.3")
    ap.add_argument("--long", type=int, default=0, metavar="TOKENS", help="长上下文测试的大致 token 数")
    ap.add_argument("--bench", type=int, default=0, metavar="N", help="并发吞吐测试的并发数")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--wait", type=float, default=0, metavar="SECONDS", help="先等服务就绪，最多等这么多秒")
    a = ap.parse_args()

    c = Client(a.base.rstrip("/"), a.key, a.model, a.timeout)
    if a.wait:
        print(f"等待 {a.base} 就绪（最多 {a.wait:.0f}s）", flush=True)
        if not wait_ready(c, a.wait):
            print(f"[FAIL] {a.wait:.0f}s 内服务没有就绪", flush=True)
            sys.exit(1)
    tests = [("模型列表", t_models), ("原样输出 / 思考分离", t_echo), ("工具调用", t_tool)]
    if a.long:
        tests.append((f"长上下文（约 {a.long} token）", lambda cl: t_long(cl, a.long)))
    if a.bench:
        tests.append((f"{a.bench} 并发吞吐", lambda cl: t_bench(cl, a.bench)))

    print(f"目标 {a.base}  模型 {a.model}\n")
    nfail = 0
    for name, fn in tests:
        try:
            passed, msg = fn(c)
        except Exception as e:  # 连接失败、超时、HTTP 错误等
            passed, msg = False, f"{type(e).__name__}: {e}"
        nfail += not passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}：{msg}", flush=True)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
