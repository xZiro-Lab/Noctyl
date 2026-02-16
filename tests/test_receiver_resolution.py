"""Direct tests for receiver_resolution: build_alias_map and resolve_receiver."""

import ast

from noctyl.ingestion.receiver_resolution import build_alias_map, resolve_receiver


def _parse(source: str) -> ast.AST:
    return ast.parse(source)


# ── build_alias_map ──────────────────────────────────────────────────────


def test_build_alias_map_direct_alias():
    """Simple x = g maps x -> g when g is a root."""
    tree = _parse("x = g")
    alias_map = build_alias_map(tree, roots={"g"})
    assert alias_map == {"x": "g"}


def test_build_alias_map_chained_alias():
    """Chained aliases: x = g, y = x both resolve to g."""
    tree = _parse("x = g\ny = x")
    alias_map = build_alias_map(tree, roots={"g"})
    assert alias_map == {"x": "g", "y": "g"}


def test_build_alias_map_triple_chain():
    """Three-level chain: a = g, b = a, c = b all resolve to g."""
    tree = _parse("a = g\nb = a\nc = b")
    alias_map = build_alias_map(tree, roots={"g"})
    assert alias_map["a"] == "g"
    assert alias_map["b"] == "g"
    assert alias_map["c"] == "g"


def test_build_alias_map_non_root_ignored():
    """Assignment from non-root variable is not aliased."""
    tree = _parse("x = z")
    alias_map = build_alias_map(tree, roots={"g"})
    assert alias_map == {}


def test_build_alias_map_multiple_roots():
    """Two roots: aliases resolve to their respective root."""
    tree = _parse("x = g1\ny = g2")
    alias_map = build_alias_map(tree, roots={"g1", "g2"})
    assert alias_map == {"x": "g1", "y": "g2"}


def test_build_alias_map_overwrites_later():
    """Later assignment overwrites earlier alias."""
    tree = _parse("x = g1\nx = g2")
    alias_map = build_alias_map(tree, roots={"g1", "g2"})
    assert alias_map["x"] == "g2"


def test_build_alias_map_non_name_value_skipped():
    """x = func() is not a simple Name assignment — skipped."""
    tree = _parse("x = func()")
    alias_map = build_alias_map(tree, roots={"func"})
    assert alias_map == {}


def test_build_alias_map_tuple_unpack_skipped():
    """Tuple unpacking assignment is not handled (targets > 1)."""
    tree = _parse("a, b = g, h")
    alias_map = build_alias_map(tree, roots={"g", "h"})
    assert alias_map == {}


def test_build_alias_map_empty_source():
    """Empty source -> empty alias map."""
    tree = _parse("")
    alias_map = build_alias_map(tree, roots={"g"})
    assert alias_map == {}


def test_build_alias_map_empty_roots():
    """No roots -> nothing to alias."""
    tree = _parse("x = g")
    alias_map = build_alias_map(tree, roots=set())
    assert alias_map == {}


# ── resolve_receiver ─────────────────────────────────────────────────────


def _name_node(name: str) -> ast.Name:
    """Create an ast.Name node."""
    return ast.Name(id=name, ctx=ast.Load())


def test_resolve_receiver_direct_root():
    """Receiver that is a root resolves to itself."""
    assert resolve_receiver(_name_node("g"), roots={"g"}, alias_map={}) == "g"


def test_resolve_receiver_alias():
    """Receiver that is an alias resolves to the root."""
    assert resolve_receiver(_name_node("x"), roots={"g"}, alias_map={"x": "g"}) == "g"


def test_resolve_receiver_unknown_name():
    """Unknown name resolves to None."""
    assert resolve_receiver(_name_node("z"), roots={"g"}, alias_map={"x": "g"}) is None


def test_resolve_receiver_non_name_node():
    """Non-Name AST node (e.g. Attribute) resolves to None."""
    attr = ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="graph", ctx=ast.Load())
    assert resolve_receiver(attr, roots={"graph"}, alias_map={}) is None


def test_resolve_receiver_root_takes_precedence_over_alias():
    """If name is both a root and in alias_map, root wins (checked first)."""
    assert resolve_receiver(_name_node("g"), roots={"g"}, alias_map={"g": "other"}) == "g"
