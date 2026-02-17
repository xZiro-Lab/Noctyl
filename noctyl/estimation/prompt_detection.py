"""
AST-based static prompt-size detection for node callables.
Extracts string literals, f-strings, and template constants to estimate base_prompt_tokens.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from noctyl.estimation.data_model import ModelProfile, NodeTokenSignature
from noctyl.graph.graph import WorkflowGraph


@dataclass(frozen=True)
class PromptFragment:
    """A detected prompt string fragment with token estimate."""

    text: str  # Extracted string text
    token_estimate: int  # Estimated tokens from this fragment
    symbolic: bool  # True if fragment contains dynamic/unresolvable parts


def estimate_tokens_from_string(text: str) -> int:
    """
    Simple heuristic token estimation: len(text) / 4 (chars-to-tokens approximation).
    
    No external tokenizer dependency (stdlib-only).
    Minimum 1 token even for empty strings.
    """
    return max(1, len(text) // 4)


def _collect_local_functions(tree: ast.AST) -> set[str]:
    """Names of functions defined at module level (reused from node_annotation pattern)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
    return names


def _collect_imported_names(tree: ast.AST) -> set[str]:
    """Names that are imported (module-level) (reused from node_annotation pattern)."""
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


def _extract_strings_from_node(node: ast.AST) -> list[PromptFragment]:
    """
    Recursively extract string literals from an AST node.
    Returns list of PromptFragment.
    """
    fragments: list[PromptFragment] = []
    
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        # Literal string
        text = node.value
        fragments.append(
            PromptFragment(
                text=text,
                token_estimate=estimate_tokens_from_string(text),
                symbolic=False,
            )
        )
        # Don't recurse into Constant nodes
    
    elif isinstance(node, ast.JoinedStr):
        # F-string: extract static portions, mark symbolic if dynamic parts present
        static_parts: list[str] = []
        has_dynamic = False
        
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                static_parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                has_dynamic = True
        
        if static_parts:
            text = "".join(static_parts)
            fragments.append(
                PromptFragment(
                    text=text,
                    token_estimate=estimate_tokens_from_string(text),
                    symbolic=has_dynamic,
                )
            )
        elif has_dynamic:
            # F-string with only dynamic parts
            fragments.append(
                PromptFragment(
                    text="",
                    token_estimate=0,
                    symbolic=True,
                )
            )
        # Don't recurse into JoinedStr.values (already processed)
    
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # String concatenation: recursively extract from both operands
        left_frags = _extract_strings_from_node(node.left)
        right_frags = _extract_strings_from_node(node.right)
        
        # Only combine if both sides have fragments and all are non-symbolic strings
        if left_frags and right_frags and not any(f.symbolic for f in left_frags + right_frags):
            # Combine adjacent string fragments
            combined_text = "".join(f.text for f in left_frags + right_frags)
            fragments.append(
                PromptFragment(
                    text=combined_text,
                    token_estimate=estimate_tokens_from_string(combined_text),
                    symbolic=False,
                )
            )
        else:
            # Keep fragments separate if symbolic, mixed, or only one side has fragments
            fragments.extend(left_frags)
            fragments.extend(right_frags)
        # Don't recurse into BinOp children (already processed left/right)
    
    elif isinstance(node, ast.Call):
        # Check for .format() calls on strings
        if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
            # Try to extract format string from func.value
            format_frags = _extract_strings_from_node(node.func.value)
            if format_frags:
                format_frag = format_frags[0]  # Use first fragment
                if not format_frag.symbolic:
                    # Check if args are all string literals
                    all_literal = True
                    arg_texts = []
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            arg_texts.append(arg.value)
                        else:
                            all_literal = False
                            break
                    
                    if all_literal:
                        # Combine format string with literal args
                        combined = format_frag.text + "".join(arg_texts)
                        fragments.append(
                            PromptFragment(
                                text=combined,
                                token_estimate=estimate_tokens_from_string(combined),
                                symbolic=False,
                            )
                        )
                    else:
                        # Dynamic args -> mark symbolic but keep format string estimate
                        fragments.append(
                            PromptFragment(
                                text=format_frag.text,
                                token_estimate=format_frag.token_estimate,
                                symbolic=True,
                            )
                        )
                else:
                    # Format string itself is symbolic
                    fragments.append(format_frag)
            # Don't recurse into .format() Call (already processed func.value and args)
        else:
            # Other Call nodes: recurse into children
            for child in ast.iter_child_nodes(node):
                fragments.extend(_extract_strings_from_node(child))
    
    else:
        # For other node types, recursively walk child nodes
        for child in ast.iter_child_nodes(node):
            fragments.extend(_extract_strings_from_node(child))
    
    return fragments


def detect_prompt_strings(source: str, callable_name: str) -> list[PromptFragment]:
    """
    Walk the AST of a callable body to extract prompt strings.
    
    Returns list of PromptFragment. Returns empty list if callable not found or unresolvable.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    
    # Resolve callable to AST node
    func_node: ast.FunctionDef | ast.Lambda | None = None
    
    if callable_name == "lambda":
        # Find lambda nodes (limited support)
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                func_node = node
                break
        # Lambdas are hard to analyze, return empty (will be marked symbolic)
        return []
    
    elif "." in callable_name:
        # module.func style - can't resolve body, mark symbolic
        return []
    
    elif _class_method_heuristic(callable_name):
        # Class method - can't resolve body from module-level AST
        return []
    
    else:
        # Find FunctionDef with matching name
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == callable_name:
                func_node = node
                break
    
    if func_node is None:
        return []
    
    # Extract strings from function body
    fragments: list[PromptFragment] = []
    if isinstance(func_node, ast.FunctionDef):
        for stmt in func_node.body:
            fragments.extend(_extract_strings_from_node(stmt))
    
    return fragments


def _has_input_dependency(func_node: ast.FunctionDef) -> bool:
    """
    Heuristic: check if function body references state, input, or function parameters.
    Similar to _state_interaction_from_body in node_annotation.py.
    """
    # Collect parameter names
    param_names: set[str] = set()
    for arg in func_node.args.args:
        param_names.add(arg.arg)
    
    # If function has parameters, assume input dependency (conservative)
    if param_names:
        return True
    
    # Scan body for references to state/input
    for node in ast.walk(func_node):
        if isinstance(node, ast.Name):
            name_lower = node.id.lower()
            if name_lower in ("state", "input"):
                return True
        
        # Check for state/input in attribute access
        if isinstance(node, (ast.Attribute, ast.Subscript)):
            try:
                s = ast.unparse(node)
                if "state" in s.lower() or "input" in s.lower():
                    return True
            except Exception:
                pass
    
    return False


def compute_node_token_signatures(
    wg: WorkflowGraph,
    source: str | None,
    model_profile: ModelProfile,
) -> tuple[NodeTokenSignature, ...]:
    """
    Compute NodeTokenSignature for each node in WorkflowGraph.
    
    For each node, resolves its callable in the source AST, detects prompt strings,
    and computes base_prompt_tokens. If callable unresolvable or no literals found,
    marks as symbolic with defaults.
    
    Returns sorted tuple of NodeTokenSignature (sorted by node_name).
    """
    # If no source, return all nodes as symbolic with defaults
    if source is None:
        return tuple(
            sorted(
                (
                    NodeTokenSignature(
                        node_name=n.name,
                        base_prompt_tokens=0,
                        expansion_factor=model_profile.expansion_factor,
                        input_dependency=False,
                        symbolic=True,
                    )
                    for n in wg.nodes
                ),
                key=lambda s: s.node_name,
            )
        )
    
    # Parse source to AST
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Syntax error -> all symbolic
        return tuple(
            sorted(
                (
                    NodeTokenSignature(
                        node_name=n.name,
                        base_prompt_tokens=0,
                        expansion_factor=model_profile.expansion_factor,
                        input_dependency=False,
                        symbolic=True,
                    )
                    for n in wg.nodes
                ),
                key=lambda s: s.node_name,
            )
        )
    
    # Build callable resolution maps (reuse patterns from node_annotation.py)
    local_funcs = _collect_local_functions(tree)
    imported = _collect_imported_names(tree)
    
    # Build name -> FunctionDef map
    name_to_func: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            name_to_func[node.name] = node
    
    signatures: list[NodeTokenSignature] = []
    
    for node in wg.nodes:
        callable_ref = node.callable_ref
        symbolic = False
        base_prompt_tokens = 0
        input_dependency = False
        
        # Resolve callable
        if callable_ref == "lambda":
            # Lambdas hard to analyze -> symbolic
            symbolic = True
        elif _class_method_heuristic(callable_ref):
            # Class methods -> can't analyze body -> symbolic
            symbolic = True
        elif "." in callable_ref:
            # module.func style -> check if imported
            root = callable_ref.split(".", 1)[0]
            if root in imported:
                # Imported function -> can't analyze body -> symbolic
                symbolic = True
            else:
                # Unresolvable -> symbolic
                symbolic = True
        else:
            # Name-like callable
            if callable_ref in local_funcs:
                # Local function -> can analyze
                fragments = detect_prompt_strings(source, callable_ref)
                
                if fragments:
                    # Sum token estimates from all fragments
                    base_prompt_tokens = sum(f.token_estimate for f in fragments)
                    # If any fragment is symbolic, mark whole node as symbolic
                    symbolic = any(f.symbolic for f in fragments)
                else:
                    # No strings found -> symbolic
                    symbolic = True
                
                # Determine input_dependency
                if callable_ref in name_to_func:
                    input_dependency = _has_input_dependency(name_to_func[callable_ref])
            
            elif callable_ref in imported:
                # Imported function -> can't analyze body -> symbolic
                symbolic = True
            else:
                # Unresolvable -> symbolic
                symbolic = True
        
        signatures.append(
            NodeTokenSignature(
                node_name=node.name,
                base_prompt_tokens=base_prompt_tokens,
                expansion_factor=model_profile.expansion_factor,
                input_dependency=input_dependency,
                symbolic=symbolic,
            )
        )
    
    return tuple(sorted(signatures, key=lambda s: s.node_name))
