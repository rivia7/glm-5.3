#!/usr/bin/env python3
"""节点0 的 Dynamo SGLang worker：在 dynamo.sglang 外面包一层，修正 PP 下 attention 组的归属。

用法同 python3 -m dynamo.sglang（SGLang 参数 + Dynamo 参数），由 launch_glm53_pp2.sh 在 ROUTER=dynamo 时调用。

为什么要包一层：
1. Dynamo（1.6）认为多机时 attention 组按机器切分（8 组、2 台 → 节点0 是第 0–3 组），
   节点0 只订阅这几组的 KV 事件。PP2 下每台机器都有全部 8 组（各放一半的层），SGLang 只在
   第一级流水（pp_rank 0，即节点0）发布 KV 事件和负载。不改的话，路由器只知道一半组的前缀缓存。
   这里把「本机有哪些组」改成按「每级流水几台机器」算：节点0 订阅全部 8 组；
   后面几级的机器不发布 KV 事件（节点1 跑的是原生 SGLang，用不到这一项）。
2. KV 事件里带着 token，启动脚本让 SGLang 只绑 127.0.0.1；Dynamo 遇到非通配地址会去连本机网卡 IP，
   这里改成端点是回环地址时就连回环地址。
"""
import os
from urllib.parse import urlparse

os.environ.setdefault("PYTHONHASHSEED", "0")  # 与 python -m dynamo.sglang 一致

import dynamo.sglang.capacity as _cap  # noqa: E402
import dynamo.sglang.publisher as _pub  # noqa: E402

_orig_local_dp_rank_bounds = _cap.local_dp_rank_bounds
_orig_publishes_kv_events = _cap.publishes_kv_events


def _int_attr(server_args, name, default):
    v = getattr(server_args, name, default)
    return v if isinstance(v, int) and v > 0 else default


def _nodes_per_stage(server_args):
    nnodes = _int_attr(server_args, "nnodes", 1)
    pp_size = _int_attr(server_args, "pp_size", 1)
    return max(nnodes // pp_size, 1)


def local_dp_rank_bounds(server_args):
    dp_size, dp_attention = _cap.sglang_dp_layout(server_args)
    if _int_attr(server_args, "pp_size", 1) > 1 and dp_attention and dp_size > 1:
        per_stage = _nodes_per_stage(server_args)
        node_rank = getattr(server_args, "node_rank", 0) or 0
        local = dp_size // per_stage
        start = (node_rank % per_stage) * local
        return start, start + local
    return _orig_local_dp_rank_bounds(server_args)


def publishes_kv_events(server_args):
    if _int_attr(server_args, "pp_size", 1) > 1:
        # SGLang 只让 pp_rank 0 的调度进程发布 KV 事件（sglang.srt.disaggregation.kv_events.is_kv_publisher_rank）
        node_rank = getattr(server_args, "node_rank", 0) or 0
        return node_rank < _nodes_per_stage(server_args)
    return _orig_publishes_kv_events(server_args)


_orig_kv_event_connect_ip = _pub.kv_event_connect_ip


def kv_event_connect_ip(endpoint):
    if urlparse(endpoint).hostname in ("127.0.0.1", "localhost"):
        return "127.0.0.1"
    return _orig_kv_event_connect_ip(endpoint)


# publisher.py 是 from ... import 进来的，两处都要换
_cap.local_dp_rank_bounds = local_dp_rank_bounds
_cap.publishes_kv_events = publishes_kv_events
_pub.local_dp_rank_bounds = local_dp_rank_bounds
_pub.publishes_kv_events = publishes_kv_events
_pub.kv_event_connect_ip = kv_event_connect_ip

if __name__ == "__main__":  # SGLang 用 spawn 起子进程，子进程会重新导入本文件，不能在这之外启动
    from dynamo.sglang.main import main

    main()
