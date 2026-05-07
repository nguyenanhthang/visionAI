"""
core/flow_graph.py
Mô hình dữ liệu pipeline: NodeInstance, Connection, FlowGraph
"""
from __future__ import annotations
import uuid
import json
import copy
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from core.tool_registry import TOOL_BY_ID, ToolDef, ParamDef


class NodeInstance:
    def __init__(self, tool_id: str, pos_x: float = 100, pos_y: float = 100):
        self.node_id: str = str(uuid.uuid4())[:8]
        self.tool_id: str = tool_id
        self.pos_x: float = pos_x
        self.pos_y: float = pos_y

        tool: ToolDef = TOOL_BY_ID[tool_id]
        # Init params from defaults
        self.params: Dict[str, Any] = {p.name: p.default for p in tool.params}

        # Runtime state
        self.outputs: Dict[str, Any] = {}
        self.status: str = "idle"   # idle | running | pass | fail | error
        self.error_msg: str = ""

    @property
    def tool(self) -> ToolDef:
        return TOOL_BY_ID[self.tool_id]

    def to_dict(self) -> dict:
        import numpy as np
        # Strip non-serializable params (numpy arrays, PatMaxModel, etc.)
        safe_params = {}
        for k, v in self.params.items():
            if isinstance(v, np.ndarray):
                continue   # skip large arrays — reload from model file
            if hasattr(v, '__class__') and v.__class__.__name__ in (
                    'PatMaxModel',):
                continue   # skip engine objects
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
    def execute(self, progress_cb=None) -> Dict[str, Any]:
        order = self.topo_order()
        total = len(order)
        results = {}

        for i, nid in enumerate(order):
            node = self.nodes[nid]
            node.status = "running"

            # Collect inputs from connected outputs
            inputs: Dict[str, Any] = {}
            for c in self.connections:
                if c.dst_id == nid:
                    src = self.nodes.get(c.src_id)
                    if src and c.src_port in src.outputs:
                        inputs[c.dst_port] = src.outputs[c.src_port]

            # Set defaults for unconnected optional inputs
            for port in node.tool.inputs:
                if port.name not in inputs:
                    inputs[port.name] = port.default

            try:
                out = node.tool.process_fn(inputs, node.params)
                node.outputs = out if out else {}
                node.status = "pass"
                # Detect pass/fail from output
                if "pass" in node.outputs:
                    node.status = "pass" if node.outputs["pass"] else "fail"
            except Exception as e:
                node.outputs = {}
                node.status = "error"
                node.error_msg = str(e)

            results[nid] = {"status": node.status, "outputs": node.outputs}
            if progress_cb:
                progress_cb(int((i + 1) / total * 100))

        return results

    def reset_status(self):
        for node in self.nodes.values():
            node.status = "idle"
            node.outputs = {}

    # ── Serialization ─────────────────────────────
    def to_dict(self) -> dict:
        return {
            "nodes":       [n.to_dict() for n in self.nodes.values()],
            "connections": [c.to_dict() for c in self.connections],
        }

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
        return g

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False,
                      default=_json_safe)

def _json_safe(obj):
    """Custom JSON serializer — bỏ qua numpy arrays và objects không serializable."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return f"<numpy array {obj.shape} {obj.dtype}>"
    if hasattr(obj, '__class__'):
        return f"<{obj.__class__.__name__}>"
    raise TypeError(f"Not serializable: {type(obj)}")

    @classmethod
    def load(cls, path: str) -> "FlowGraph":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
