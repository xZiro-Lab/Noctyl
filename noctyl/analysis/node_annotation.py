"""
Node semantic annotation: callable origin, state interaction, role heuristic.
Uses source and file_path when provided; otherwise returns unknown for all.
"""

from __future__ import annotations

import ast

from noctyl.graph.execution_model import NodeAnnotation
from noctyl.graph.graph import WorkflowGraph

# Literals aligned with execution_model
ORIGIN_LOCAL = "local_function"
ORIGIN_IMPORTED = "imported_function"
ORIGIN_CLASS_METHOD = "class_method"
ORIGIN_LAMBDA = "lambda"
ORIGIN_UNKNOWN = "unknown"

STATE_PURE = "pure"
STATE_READ_ONLY = "read_only"
STATE_MUTATES = "mutates_state"
STATE_UNKNOWN = "unknown"

ROLE_LLM = "llm_like"
ROLE_TOOL = "tool_like"
ROLE_CONTROL = "control_node"
ROLE_UNKNOWN = "unknown"


def _collect_local_functions(tree: ast.AST) -> set[str]:
    """Names of functions defined at module level."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
    return names


def _collect_imported_names(tree: ast.AST) -> set[str]:
    """Names that are imported (module-level)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _class_method_heuristic(callable_ref: str) -> bool:
    """Attribute access like self.xxx/cls.xxx -> class_method."""
    return callable_ref.startswith("self.") or callable_ref.startswith("cls.")


def _state_interaction_from_body(node: ast.FunctionDef) -> str:
    """Heuristic: scan body for state read/write."""
    mutates = False
    reads = False
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                try:
                    s = ast.unparse(t)
                    if "state" in s.lower():
                        mutates = True
                except Exception:
                    pass
        if isinstance(n, ast.Subscript) or isinstance(n, ast.Attribute):
            try:
                s = ast.unparse(n)
                if "state" in s.lower():
                    reads = True
            except Exception:
                pass
        if isinstance(n, (ast.Call, ast.Attribute)):
            try:
                s = ast.unparse(n)
                if "state" in s.lower():
                    reads = True
            except Exception:
                pass
    if mutates:
        return STATE_MUTATES
    if reads:
        return STATE_READ_ONLY
    return STATE_PURE


def _role_heuristic(node_name: str, callable_ref: str) -> str:
    """Name/symbol hints for role."""
    ref_lower = callable_ref.lower()
    name_lower = node_name.lower()
    if "llm" in ref_lower or "llm" in name_lower or "chat" in ref_lower or "invoke" in ref_lower:
        return ROLE_LLM
    if "tool" in ref_lower or "tool" in name_lower:
        return ROLE_TOOL
    if "route" in ref_lower or "router" in ref_lower or "conditional" in ref_lower or "control" in ref_lower:
        return ROLE_CONTROL
    return ROLE_UNKNOWN


def compute_node_annotations(
    wg: WorkflowGraph,
    source: str | None = None,
    file_path: str | None = None,
) -> tuple[NodeAnnotation, ...]:
    """
    One NodeAnnotation per workflow node, sorted by node_name.
    When source is None, all fields are unknown.
    """
    result: list[NodeAnnotation] = []
    for n in wg.nodes:
        result.append(
            NodeAnnotation(
                node_name=n.name,
                origin=ORIGIN_UNKNOWN,
                state_interaction=STATE_UNKNOWN,
                role=ROLE_UNKNOWN,
            )
        )

    if not source:
        return tuple(sorted(result, key=lambda a: a.node_name))

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return tuple(sorted(result, key=lambda a: a.node_name))

    local_funcs = _collect_local_functions(tree)
    imported = _collect_imported_names(tree)

    # Build name -> FunctionDef for state_interaction
    name_to_func: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            name_to_func[node.name] = node

    annotations_by_name: dict[str, NodeAnnotation] = {}
    for n in wg.nodes:
        origin = ORIGIN_UNKNOWN
        state = STATE_UNKNOWN
        role = _role_heuristic(n.name, n.callable_ref)

        if n.callable_ref == "lambda":
            origin = ORIGIN_LAMBDA
        elif _class_method_heuristic(n.callable_ref):
            origin = ORIGIN_CLASS_METHOD
        else:
            if "." in n.callable_ref:
                # module.func style
                root = n.callable_ref.split(".", 1)[0]
                if root in imported:
                    origin = ORIGIN_IMPORTED
            else:
                # Name-like
                sym = n.callable_ref
                if sym in local_funcs:
                    origin = ORIGIN_LOCAL
                    if sym in name_to_func:
                        state = _state_interaction_from_body(name_to_func[sym])
                elif sym in imported:
                    origin = ORIGIN_IMPORTED

        annotations_by_name[n.name] = NodeAnnotation(
            node_name=n.name,
            origin=origin,
            state_interaction=state,
            role=role,
        )

    out = [
        annotations_by_name.get(no.name, NodeAnnotation(no.name, ORIGIN_UNKNOWN, STATE_UNKNOWN, ROLE_UNKNOWN))
        for no in wg.nodes
    ]
    return tuple(sorted(out, key=lambda a: a.node_name))
