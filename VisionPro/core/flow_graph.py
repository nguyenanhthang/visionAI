"""
core/flow_graph.py
Mô hình dữ liệu pipeline: NodeInstance, Connection, FlowGraph
"""
from __future__ import annotations
import uuid
import json
import copy
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from core.tool_registry import TOOL_BY_ID, ToolDef, ParamDef


# Default hidden output ports cho từng tool_id (chỉ áp dụng khi tạo node MỚI
# — node load từ file giữ nguyên _hidden_outputs đã save). Mục đích: tools có
# nhiều output scalar (blob, find_circle…) chỉ hiện những port "primary" để
# node gọn; user vẫn unhide qua dialog "👁 Manage Output Ports".
_DEFAULT_HIDDEN_OUTPUTS: Dict[str, List[str]] = {
    "blob": ["blobs", "centroids", "cx", "cy",
             "bbox_w", "bbox_h", "angle"],
}


class NodeInstance:
    def __init__(self, tool_id: str, pos_x: float = 100, pos_y: float = 100):
        self.node_id: str = str(uuid.uuid4())[:8]
        self.tool_id: str = tool_id
        self.pos_x: float = pos_x
        self.pos_y: float = pos_y

        tool: ToolDef = TOOL_BY_ID[tool_id]
        # Init params from defaults
        self.params: Dict[str, Any] = {p.name: p.default for p in tool.params}
        hidden = _DEFAULT_HIDDEN_OUTPUTS.get(tool_id)
        if hidden:
            self.params["_hidden_outputs"] = list(hidden)

        # Runtime state
        self.outputs: Dict[str, Any] = {}
        self.status: str = "idle"   # idle | running | pass | fail | error
        self.error_msg: str = ""
        self.last_run_ms: float = 0.0   # wall-clock của lần process_fn gần nhất

    @property
    def tool(self) -> ToolDef:
        return TOOL_BY_ID[self.tool_id]

    def to_dict(self) -> dict:
        import numpy as np
        # Strip non-serializable params (numpy arrays, etc.). PatMaxModel
        # được nhúng dưới dạng dict JSON-safe (meta + npz base64) để pipeline
        # tự chứa model — mang .aoi sang máy khác là chạy.
        safe_params = {}
        for k, v in self.params.items():
            if isinstance(v, np.ndarray):
                continue   # skip large arrays — không thuộc model
            cls_name = getattr(getattr(v, '__class__', None), '__name__', '')
            if cls_name == 'PatMaxModel':
                try:
                    from core.patmax_engine import model_to_serializable
                    safe_params[k] = model_to_serializable(v)
                except Exception as e:
                    print(f"[FlowGraph] embed PatMaxModel '{k}' failed: {e}")
                continue
            if isinstance(v, list) and v and \
                    getattr(getattr(v[0], '__class__', None), '__name__', '') == 'PatMaxModel':
                try:
                    from core.patmax_engine import model_to_serializable
                    safe_params[k] = [model_to_serializable(m) for m in v]
                except Exception as e:
                    print(f"[FlowGraph] embed PatMaxModel list '{k}' failed: {e}")
                continue
            if isinstance(v, (int, float, bool, str, list, dict, tuple, type(None))):
                safe_params[k] = v
            elif isinstance(v, tuple):
                safe_params[k] = list(v)
            # else skip silently
        return {
            "node_id":  self.node_id,
            "tool_id":  self.tool_id,
            "pos_x":    self.pos_x,
            "pos_y":    self.pos_y,
            "params":   safe_params,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NodeInstance":
        n = cls(d["tool_id"], d["pos_x"], d["pos_y"])
        n.node_id = d["node_id"]
        n.params  = d["params"]
        # Reconstruct embedded PatMaxModel(s). Fallback: load từ
        # `_patmax_model_file` (đường dẫn) nếu không có embed.
        try:
            from core.patmax_engine import model_from_serializable, load_model
        except Exception:
            model_from_serializable = None
            load_model = None
        for key in ("_patmax_model", "_patmax_models"):
            val = n.params.get(key)
            if isinstance(val, dict) and val.get("_patmax_embedded"):
                m = model_from_serializable(val) if model_from_serializable else None
                if m is not None:
                    n.params[key] = m
            elif isinstance(val, list) and val and isinstance(val[0], dict) \
                    and val[0].get("_patmax_embedded"):
                if model_from_serializable:
                    n.params[key] = [model_from_serializable(x) for x in val
                                      if isinstance(x, dict)]
        # Fallback path-based load nếu chưa có model object (.aoi cũ
        # không có embed, chỉ có `_patmax_model_file`).
        cur = n.params.get("_patmax_model")
        has_model = getattr(getattr(cur, '__class__', None),
                            '__name__', '') == 'PatMaxModel'
        if load_model and not has_model:
            path = n.params.get("_patmax_model_file")
            if isinstance(path, str) and path:
                try:
                    import os as _os
                    if _os.path.exists(_os.path.splitext(path)[0] + ".json"):
                        m = load_model(path)
                        if m is not None:
                            n.params["_patmax_model"] = m
                            print(f"[FlowGraph] auto-loaded PatMax model: {path}")
                except Exception as e:
                    print(f"[FlowGraph] fallback load_model failed: {e}")
        return n


class Connection:
    """src_node:src_port → dst_node:dst_port"""
    def __init__(self, src_id: str, src_port: str, dst_id: str, dst_port: str):
        self.conn_id  = str(uuid.uuid4())[:8]
        self.src_id   = src_id
        self.src_port = src_port
        self.dst_id   = dst_id
        self.dst_port = dst_port

    def to_dict(self) -> dict:
        return {"conn_id": self.conn_id,
                "src_id": self.src_id, "src_port": self.src_port,
                "dst_id": self.dst_id, "dst_port": self.dst_port}

    @classmethod
    def from_dict(cls, d: dict) -> "Connection":
        c = cls(d["src_id"], d["src_port"], d["dst_id"], d["dst_port"])
        c.conn_id = d["conn_id"]
        return c


class FlowGraph:
    def __init__(self):
        self.nodes: Dict[str, NodeInstance] = {}
        self.connections: List[Connection] = []
        # UI state persisted với pipeline file — vd `selected_overlays`
        # (Results dropdown trong ImageViewerPanel). Để ngoài node để
        # không gắn cứng vào schema NodeInstance; UI code đọc/ghi
        # trực tiếp các key cần.
        self.ui_state: Dict[str, Any] = {}

    # ── Node CRUD ──────────────────────────────────
    def add_node(self, tool_id: str, x: float, y: float) -> NodeInstance:
        node = NodeInstance(tool_id, x, y)
        self.nodes[node.node_id] = node
        return node

    def remove_node(self, node_id: str):
        self.nodes.pop(node_id, None)
        self.connections = [c for c in self.connections
                            if c.src_id != node_id and c.dst_id != node_id]

    # ── Connection CRUD ────────────────────────────
    def add_connection(self, src_id, src_port, dst_id, dst_port) -> Optional[Connection]:
        # Prevent duplicate to same dst input port (one input per port)
        self.connections = [c for c in self.connections
                            if not (c.dst_id == dst_id and c.dst_port == dst_port)]
        if src_id == dst_id:
            return None
        conn = Connection(src_id, src_port, dst_id, dst_port)
        self.connections.append(conn)
        return conn

    def remove_connection(self, conn_id: str):
        self.connections = [c for c in self.connections if c.conn_id != conn_id]

    def connections_for_node(self, node_id: str) -> List[Connection]:
        return [c for c in self.connections
                if c.src_id == node_id or c.dst_id == node_id]

    # Tên port được auto-bind từ upstream khi chưa wired. Giảm spaghetti
    # cho 2 channel hay dùng nhất: ảnh (gần như mọi tool đều cần) và mask
    # (blob/threshold/morphology/find_contours/area_measure).
    _IMPLICIT_CHANNELS = ("image", "mask")

    def _implicit_sources(
        self,
        nodes: Dict[str, "NodeInstance"],
        in_edges: Dict[str, set],
    ) -> Dict[Tuple[str, str], str]:
        """Cho mỗi (node_id, channel) chưa wired, trả node_id "latest"
        cung cấp channel đó trong topo order. Hai bước:
          1. BFS upstream qua connections — ưu tiên (đảm bảo causal).
          2. Fallback: latest producer trong topo order trước node hiện
             tại (cho phép bỏ wire image dư thừa mà pipeline vẫn chạy).
        Caller phải add các implicit dep này vào in_edges/out_edges để
        parallel scheduler đợi producer xong trước consumer.
        """
        out_adj: Dict[str, set] = {nid: set() for nid in nodes}
        parent_of: Dict[str, List[str]] = {}
        for c in self.connections:
            if c.src_id in nodes and c.dst_id in nodes:
                out_adj[c.src_id].add(c.dst_id)
                parent_of.setdefault(c.dst_id, []).append(c.src_id)

        in_count = {nid: len(deps) for nid, deps in in_edges.items()}
        in_count_w = dict(in_count)
        ready = [nid for nid, cnt in in_count_w.items() if cnt == 0]
        topo: List[str] = []
        while ready:
            nid = ready.pop(0)
            topo.append(nid)
            for succ in out_adj.get(nid, ()):
                in_count_w[succ] -= 1
                if in_count_w[succ] == 0:
                    ready.append(succ)

        wired_in: set = {(c.dst_id, c.dst_port)
                         for c in self.connections
                         if c.src_id in nodes and c.dst_id in nodes}

        implicit: Dict[Tuple[str, str], str] = {}

        def bfs_upstream(nid: str, ch: str) -> Optional[str]:
            visited = {nid}
            queue: List[str] = list(parent_of.get(nid, []))
            while queue:
                cur = queue.pop(0)
                if cur in visited:
                    continue
                visited.add(cur)
                node = nodes.get(cur)
                if node and any(p.name == ch for p in node.tool.outputs):
                    return cur
                queue.extend(parent_of.get(cur, []))
            return None

        for ch in self._IMPLICIT_CHANNELS:
            latest: Optional[str] = None
            for nid in topo:
                node = nodes[nid]
                needs = any(p.name == ch for p in node.tool.inputs)
                if needs and (nid, ch) not in wired_in:
                    src = bfs_upstream(nid, ch) or latest
                    if src and src != nid:
                        implicit[(nid, ch)] = src
                if any(p.name == ch for p in node.tool.outputs):
                    latest = nid
        return implicit

    def implicit_source_for(self, nid: str, port_name: str) -> Optional[str]:
        """Wrapper public: trả node_id cung cấp implicit value cho UI
        tooltip. Dùng cùng logic execute() để hiển thị nhất quán."""
        nodes = self.nodes
        in_edges = {n: set() for n in nodes}
        for c in self.connections:
            if c.src_id in nodes and c.dst_id in nodes:
                in_edges[c.dst_id].add(c.src_id)
        return self._implicit_sources(nodes, in_edges).get((nid, port_name))

    # ── Topological sort ──────────────────────────
    def topo_order(self) -> List[str]:
        in_edges: Dict[str, set] = {nid: set() for nid in self.nodes}
        for c in self.connections:
            if c.dst_id in in_edges and c.src_id in self.nodes:
                in_edges[c.dst_id].add(c.src_id)

        order = []
        visited = set()
        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            for dep in list(in_edges.get(nid, [])):
                visit(dep)
            order.append(nid)

        for nid in self.nodes:
            visit(nid)
        return order

    # ── Execute ────────────────────────────────────
    def execute(self, progress_cb=None,
                acquire_node_id: Optional[str] = None) -> Dict[str, Any]:
        """Run pipeline. Node độc lập (không phụ thuộc nhau) chạy parallel
        qua ThreadPool — 2 branch song song (vd 2 Acquire Image) hoàn thành
        gần bằng thời gian 1 branch vì OpenCV/numpy ops nhả GIL.

        Single chain (A→B→C→D): chỉ 1 node chạy 1 lúc → speed như cũ.
        Multi-branch (Acq1→…, Acq2→…): roots chạy đồng thời, successors
        cũng đồng thời khi deps xong → tổng ≈ max(branch_durations).

        `acquire_node_id` (PLC TriggerRoute): nếu set, chỉ chạy subgraph
        downstream từ node đó (BFS theo connections) — các nhánh acquire
        khác bị skip. Cho phép 1 PLC word điều khiển nhiều camera/source
        độc lập.
        """
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        import threading

        nodes = self.nodes
        # Subgraph filter: chỉ giữ node downstream từ acquire_node_id (BFS)
        if acquire_node_id and acquire_node_id in nodes:
            keep = {acquire_node_id}
            frontier = {acquire_node_id}
            adj: Dict[str, set] = {nid: set() for nid in nodes}
            for c in self.connections:
                if c.src_id in adj and c.dst_id in nodes:
                    adj[c.src_id].add(c.dst_id)
            while frontier:
                nxt: set = set()
                for nid in frontier:
                    for d in adj.get(nid, ()):  # downstream của nid
                        if d not in keep:
                            keep.add(d)
                            nxt.add(d)
                frontier = nxt
            nodes = {nid: n for nid, n in nodes.items() if nid in keep}
        in_edges  = {nid: set() for nid in nodes}
        out_edges = {nid: set() for nid in nodes}
        for c in self.connections:
            if c.src_id in nodes and c.dst_id in nodes:
                in_edges[c.dst_id].add(c.src_id)
                out_edges[c.src_id].add(c.dst_id)

        # Implicit channel bus (image/mask): tự bind port chưa wired từ
        # producer gần nhất trong topo order. Add virtual dep để scheduler
        # đợi producer xong trước consumer — không tạo cycle vì topo
        # luôn từ trên xuống.
        implicit_sources = self._implicit_sources(nodes, in_edges)
        for (dst_nid, _ch), src_nid in implicit_sources.items():
            if src_nid != dst_nid and src_nid not in in_edges[dst_nid]:
                in_edges[dst_nid].add(src_nid)
                out_edges[src_nid].add(dst_nid)
        remaining = {nid: set(deps) for nid, deps in in_edges.items()}

        total = len(nodes)
        if total == 0:
            return {}
        results: Dict[str, Any] = {}
        done_count = 0
        progress_lock = threading.Lock()

        def run_one(nid: str):
            node = nodes[nid]
            node.status = "running"
            inputs: Dict[str, Any] = {}
            for c in self.connections:
                if c.dst_id == nid:
                    src = nodes.get(c.src_id)
                    if src and c.src_port in src.outputs:
                        inputs[c.dst_port] = src.outputs[c.src_port]
            # Implicit channel bus: lấy value từ source đã pre-compute ở
            # implicit_sources. Wire tay (đã collect ở loop trên) luôn
            # override implicit.
            for port in node.tool.inputs:
                if port.name in self._IMPLICIT_CHANNELS \
                        and inputs.get(port.name) is None:
                    src = implicit_sources.get((nid, port.name))
                    if src and src in nodes:
                        val = nodes[src].outputs.get(port.name)
                        if val is not None:
                            inputs[port.name] = val
            for port in node.tool.inputs:
                if port.name not in inputs:
                    inputs[port.name] = port.default
            t0 = time.perf_counter()
            try:
                out = node.tool.process_fn(inputs, node.params)
                node.outputs = out if out else {}
                node.status = "pass"
                if "pass" in node.outputs:
                    node.status = "pass" if node.outputs["pass"] else "fail"
            except Exception as e:
                node.outputs = {}
                node.status = "error"
                node.error_msg = str(e)
            elapsed = (time.perf_counter() - t0) * 1000.0
            node.last_run_ms = elapsed
            return nid, elapsed

        # Workers: cap ở 8 để không thrash CPU/RAM với pipeline lớn
        max_workers = min(8, max(2, total))
        pool = ThreadPoolExecutor(max_workers=max_workers)
        try:
            in_flight = set()
            # Submit initial ready (root nodes)
            for nid in nodes:
                if not remaining[nid]:
                    in_flight.add(pool.submit(run_one, nid))

            while in_flight:
                done_set, in_flight = wait(in_flight,
                                            return_when=FIRST_COMPLETED)
                for fut in done_set:
                    nid, elapsed = fut.result()
                    node = nodes[nid]
                    results[nid] = {"status": node.status,
                                    "outputs": node.outputs,
                                    "elapsed_ms": elapsed}
                    with progress_lock:
                        done_count += 1
                        if progress_cb:
                            progress_cb(int(done_count / total * 100))
                    # Release successors whose deps are all completed
                    for succ in out_edges.get(nid, ()):
                        remaining[succ].discard(nid)
                        if not remaining[succ]:
                            in_flight.add(pool.submit(run_one, succ))
        finally:
            pool.shutdown(wait=True)
        return results

    def reset_status(self):
        for node in self.nodes.values():
            node.status = "idle"
            node.outputs = {}

    # ── Serialization ─────────────────────────────
    def to_dict(self) -> dict:
        out = {
            "nodes":       [n.to_dict() for n in self.nodes.values()],
            "connections": [c.to_dict() for c in self.connections],
        }
        if self.ui_state:
            out["ui_state"] = self.ui_state
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "FlowGraph":
        g = cls()
        for nd in d.get("nodes", []):
            if nd["tool_id"] in TOOL_BY_ID:
                node = NodeInstance.from_dict(nd)
                g.nodes[node.node_id] = node
        for cd in d.get("connections", []):
            try:
                g.connections.append(Connection.from_dict(cd))
            except Exception:
                pass
        ui = d.get("ui_state")
        if isinstance(ui, dict):
            g.ui_state = ui
        return g

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False,
                      default=_json_safe)

    @classmethod
    def load(cls, path: str) -> "FlowGraph":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _json_safe(obj):
    """Custom JSON serializer — bỏ qua numpy arrays và objects không serializable."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return f"<numpy array {obj.shape} {obj.dtype}>"
    if hasattr(obj, '__class__'):
        return f"<{obj.__class__.__name__}>"
    raise TypeError(f"Not serializable: {type(obj)}")
