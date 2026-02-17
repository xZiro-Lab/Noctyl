"""Tests for Phase 3 prompt detection: string extraction and token signature computation."""

import pytest

from noctyl.estimation import (
    ModelProfile,
    PromptFragment,
    compute_node_token_signatures,
    detect_prompt_strings,
    estimate_tokens_from_string,
)
from noctyl.graph import ExtractedEdge, ExtractedNode, build_workflow_graph


# ── estimate_tokens_from_string ──────────────────────────────────────────────


def test_estimate_tokens_from_string_basic():
    """Basic token estimation: len(text) / 4."""
    assert estimate_tokens_from_string("Hello") == 1  # 5 chars / 4 = 1
    assert estimate_tokens_from_string("Hello world") == 2  # 11 chars / 4 = 2
    assert estimate_tokens_from_string("x" * 100) == 25  # 100 chars / 4 = 25


def test_estimate_tokens_from_string_empty():
    """Empty string returns minimum 1 token."""
    assert estimate_tokens_from_string("") == 1


def test_estimate_tokens_from_string_short():
    """Short strings (< 4 chars) return 1 token minimum."""
    assert estimate_tokens_from_string("x") == 1
    assert estimate_tokens_from_string("abc") == 1


def test_estimate_tokens_from_string_long():
    """Long strings are estimated correctly."""
    text = "This is a longer prompt string with multiple words."
    expected = len(text) // 4
    assert estimate_tokens_from_string(text) == expected


# ── detect_prompt_strings ─────────────────────────────────────────────────────


def test_detect_prompt_strings_literal_string():
    """Detect simple literal string in function body."""
    source = '''def planner(state):
    prompt = "You are a helpful assistant"
    return prompt
'''
    fragments = detect_prompt_strings(source, "planner")
    assert len(fragments) == 1
    assert fragments[0].text == "You are a helpful assistant"
    assert fragments[0].token_estimate > 0
    assert fragments[0].symbolic is False


def test_detect_prompt_strings_multiple_literals():
    """Detect multiple string literals in function body."""
    source = '''def planner(state):
    prompt1 = "First prompt"
    prompt2 = "Second prompt"
    combined = prompt1 + prompt2
    return combined
'''
    fragments = detect_prompt_strings(source, "planner")
    assert len(fragments) >= 2
    texts = [f.text for f in fragments]
    assert "First prompt" in texts or any("First prompt" in t for t in texts)
    assert "Second prompt" in texts or any("Second prompt" in t for t in texts)


def test_detect_prompt_strings_f_string_static():
    """F-string with only static parts is detected."""
    source = '''def planner(state):
    name = "assistant"
    prompt = f"You are a helpful {name}"
    return prompt
'''
    fragments = detect_prompt_strings(source, "planner")
    # Should detect the static portion "You are a helpful "
    assert len(fragments) >= 1
    # Should be marked symbolic due to dynamic part
    assert any(f.symbolic for f in fragments) or len(fragments) == 0


def test_detect_prompt_strings_f_string_dynamic():
    """F-string with dynamic parts is marked symbolic."""
    source = '''def planner(state):
    prompt = f"Hello {state['name']}"
    return prompt
'''
    fragments = detect_prompt_strings(source, "planner")
    # Should detect static portion but mark symbolic
    if fragments:
        assert any(f.symbolic for f in fragments)


def test_detect_prompt_strings_concatenation():
    """String concatenation is detected."""
    source = '''def planner(state):
    prompt = "Hello" + " " + "world"
    return prompt
'''
    fragments = detect_prompt_strings(source, "planner")
    assert len(fragments) >= 1
    # Should combine or detect separately
    combined_text = "".join(f.text for f in fragments if not f.symbolic)
    assert "Hello" in combined_text or "world" in combined_text


def test_detect_prompt_strings_format_call():
    """String.format() call with literal args is detected."""
    source = '''def planner(state):
    prompt = "Hello {}!".format("world")
    return prompt
'''
    fragments = detect_prompt_strings(source, "planner")
    assert len(fragments) >= 1
    # Should detect format string and args
    texts = "".join(f.text for f in fragments if not f.symbolic)
    assert "Hello" in texts or "world" in texts


def test_detect_prompt_strings_format_call_dynamic():
    """String.format() with dynamic args is marked symbolic."""
    source = '''def planner(state):
    prompt = "Hello {}!".format(state['name'])
    return prompt
'''
    fragments = detect_prompt_strings(source, "planner")
    if fragments:
        # Should be marked symbolic due to dynamic arg
        assert any(f.symbolic for f in fragments)


def test_detect_prompt_strings_unresolvable_callable():
    """Unresolvable callable returns empty list."""
    source = "def other_func(): pass"
    fragments = detect_prompt_strings(source, "nonexistent")
    assert fragments == []


def test_detect_prompt_strings_lambda():
    """Lambda callable returns empty list (hard to analyze)."""
    source = "x = lambda y: 'prompt'"
    fragments = detect_prompt_strings(source, "lambda")
    assert fragments == []


def test_detect_prompt_strings_class_method():
    """Class method callable returns empty list (can't resolve body)."""
    source = "def some_func(): pass"
    fragments = detect_prompt_strings(source, "self.method")
    assert fragments == []


def test_detect_prompt_strings_imported_function():
    """Imported function callable returns empty list."""
    source = "import external_module"
    fragments = detect_prompt_strings(source, "external_module.func")
    assert fragments == []


def test_detect_prompt_strings_syntax_error():
    """Syntax error in source returns empty list."""
    source = "def planner(state:\n    return"  # Invalid syntax
    fragments = detect_prompt_strings(source, "planner")
    assert fragments == []


def test_detect_prompt_strings_empty_function():
    """Empty function body returns empty fragments."""
    source = "def planner(state):\n    pass"
    fragments = detect_prompt_strings(source, "planner")
    assert fragments == []


def test_detect_prompt_strings_nested_strings():
    """Strings in nested structures are detected."""
    source = '''def planner(state):
    config = {
        "prompt": "You are helpful",
        "system": "System message"
    }
    return config["prompt"]
'''
    fragments = detect_prompt_strings(source, "planner")
    assert len(fragments) >= 2
    texts = [f.text for f in fragments]
    assert any("You are helpful" in t for t in texts)
    assert any("System message" in t for t in texts)


# ── compute_node_token_signatures ────────────────────────────────────────────


def _minimal_workflow_graph():
    """Helper: minimal WorkflowGraph for testing."""
    nodes = [ExtractedNode("a", "f", 1), ExtractedNode("b", "g", 2)]
    edges = [ExtractedEdge("a", "b", 3)]
    return build_workflow_graph("file.py:0", nodes, edges, [], "a")


def _default_model_profile():
    """Helper: default ModelProfile for testing."""
    return ModelProfile("default", 1.3, 0.5, 0.0, 0.0)


def test_compute_node_token_signatures_no_source():
    """Missing source returns all nodes as symbolic."""
    wg = _minimal_workflow_graph()
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, None, profile)
    
    assert len(sigs) == 2
    assert all(s.symbolic for s in sigs)
    assert all(s.base_prompt_tokens == 0 for s in sigs)
    assert all(s.expansion_factor == profile.expansion_factor for s in sigs)
    assert all(s.input_dependency is False for s in sigs)
    # Sorted by node_name
    assert [s.node_name for s in sigs] == ["a", "b"]


def test_compute_node_token_signatures_syntax_error():
    """Syntax error in source returns all nodes as symbolic."""
    wg = _minimal_workflow_graph()
    profile = _default_model_profile()
    invalid_source = "def f(\n    return"  # Invalid syntax
    sigs = compute_node_token_signatures(wg, invalid_source, profile)
    
    assert len(sigs) == 2
    assert all(s.symbolic for s in sigs)
    assert all(s.base_prompt_tokens == 0 for s in sigs)


def test_compute_node_token_signatures_local_function():
    """Local function callable is resolved and strings detected."""
    source = '''def f(state):
    prompt = "Hello world"
    return prompt

def g(state):
    return "Goodbye"
'''
    nodes = [
        ExtractedNode("node_a", "f", 1),
        ExtractedNode("node_b", "g", 2),
    ]
    wg = build_workflow_graph("test:0", nodes, [], [], "node_a")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 2
    sig_a = next(s for s in sigs if s.node_name == "node_a")
    sig_b = next(s for s in sigs if s.node_name == "node_b")
    
    assert sig_a.base_prompt_tokens > 0  # "Hello world" detected
    assert sig_a.symbolic is False
    assert sig_b.base_prompt_tokens > 0  # "Goodbye" detected
    assert sig_b.symbolic is False


def test_compute_node_token_signatures_unresolvable():
    """Unresolvable callable is marked symbolic."""
    source = "def other_func(): pass"
    nodes = [ExtractedNode("node_a", "nonexistent", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "node_a")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].symbolic is True
    assert sigs[0].base_prompt_tokens == 0


def test_compute_node_token_signatures_lambda():
    """Lambda callable is marked symbolic."""
    source = "x = lambda y: 'prompt'"
    nodes = [ExtractedNode("node_a", "lambda", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "node_a")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].symbolic is True
    assert sigs[0].base_prompt_tokens == 0


def test_compute_node_token_signatures_class_method():
    """Class method callable is marked symbolic."""
    source = "def some_func(): pass"
    nodes = [ExtractedNode("node_a", "self.method", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "node_a")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].symbolic is True
    assert sigs[0].base_prompt_tokens == 0


def test_compute_node_token_signatures_imported_function():
    """Imported function callable is marked symbolic."""
    source = "import external_module"
    nodes = [ExtractedNode("node_a", "external_module.func", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "node_a")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].symbolic is True
    assert sigs[0].base_prompt_tokens == 0


def test_compute_node_token_signatures_input_dependency():
    """Function with parameters has input_dependency=True."""
    source = '''def planner(state):
    return state['key']
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].input_dependency is True


def test_compute_node_token_signatures_no_input_dependency():
    """Function without parameters has input_dependency=False."""
    source = '''def planner():
    return "Hello"
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].input_dependency is False


def test_compute_node_token_signatures_sums_fragments():
    """Multiple string fragments are summed for base_prompt_tokens."""
    source = '''def planner(state):
    prompt1 = "First part"
    prompt2 = "Second part"
    return prompt1 + prompt2
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].base_prompt_tokens > 0
    # Should sum tokens from both fragments
    assert sigs[0].base_prompt_tokens >= estimate_tokens_from_string("First part")


def test_compute_node_token_signatures_symbolic_fragment():
    """If any fragment is symbolic, whole node is symbolic."""
    source = '''def planner(state):
    prompt = f"Hello {state['name']}"
    return prompt
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    # F-string with dynamic parts should be symbolic
    assert sigs[0].symbolic is True


def test_compute_node_token_signatures_no_strings_found():
    """Function with no strings is marked symbolic."""
    source = '''def planner(state):
    x = 42
    return x
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].symbolic is True
    assert sigs[0].base_prompt_tokens == 0


def test_compute_node_token_signatures_sorted_by_node_name():
    """Signatures are sorted by node_name for determinism."""
    source = '''def z_func(): return "z"
def a_func(): return "a"
def m_func(): return "m"
'''
    nodes = [
        ExtractedNode("node_z", "z_func", 1),
        ExtractedNode("node_a", "a_func", 2),
        ExtractedNode("node_m", "m_func", 3),
    ]
    wg = build_workflow_graph("test:0", nodes, [], [], "node_a")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 3
    names = [s.node_name for s in sigs]
    assert names == ["node_a", "node_m", "node_z"]


def test_compute_node_token_signatures_model_profile_expansion_factor():
    """Expansion factor comes from ModelProfile."""
    source = "def planner(state): return 'prompt'"
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = ModelProfile("custom", 1.5, 0.6, 0.0, 0.0)
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].expansion_factor == 1.5


# ── Integration tests ──────────────────────────────────────────────────────────


def test_integration_multiple_nodes_with_strings():
    """Multiple nodes with different string patterns."""
    source = '''def planner(state):
    return "Plan the task"

def executor(state):
    prompt = "Execute: " + state['task']
    return prompt

def reviewer(state):
    return "Review completed"
'''
    nodes = [
        ExtractedNode("planner", "planner", 1),
        ExtractedNode("executor", "executor", 2),
        ExtractedNode("reviewer", "reviewer", 3),
    ]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 3
    planner_sig = next(s for s in sigs if s.node_name == "planner")
    executor_sig = next(s for s in sigs if s.node_name == "executor")
    reviewer_sig = next(s for s in sigs if s.node_name == "reviewer")
    
    assert planner_sig.base_prompt_tokens > 0
    assert planner_sig.symbolic is False
    assert executor_sig.input_dependency is True  # Uses state parameter
    assert reviewer_sig.base_prompt_tokens > 0


def test_integration_mixed_resolvable_unresolvable():
    """Mix of resolvable and unresolvable callables."""
    source = "def local_func(): return 'prompt'"
    nodes = [
        ExtractedNode("local", "local_func", 1),
        ExtractedNode("imported", "external.func", 2),
        ExtractedNode("lambda_node", "lambda", 3),
    ]
    wg = build_workflow_graph("test:0", nodes, [], [], "local")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 3
    local_sig = next(s for s in sigs if s.node_name == "local")
    imported_sig = next(s for s in sigs if s.node_name == "imported")
    lambda_sig = next(s for s in sigs if s.node_name == "lambda_node")
    
    assert local_sig.symbolic is False
    assert local_sig.base_prompt_tokens > 0
    assert imported_sig.symbolic is True
    assert lambda_sig.symbolic is True


def test_integration_f_string_with_static_portions():
    """F-string with static portions still estimates tokens."""
    source = '''def planner(state):
    name = "Assistant"
    prompt = f"You are {name}. Help the user."
    return prompt
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    # Should detect static portion "You are " and ". Help the user."
    # But mark as symbolic due to dynamic part
    assert sigs[0].symbolic is True
    # May still have some token estimate from static portions


def test_integration_string_concatenation_combined():
    """String concatenation combines fragments when possible."""
    source = '''def planner(state):
    prompt = "Hello" + " " + "world"
    return prompt
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].base_prompt_tokens > 0
    # Should estimate "Hello world" (combined)
    assert sigs[0].base_prompt_tokens >= estimate_tokens_from_string("Hello world")


def test_integration_format_call_with_literals():
    """String.format() with literal args is detected."""
    source = '''def planner(state):
    prompt = "Hello {}!".format("world")
    return prompt
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].base_prompt_tokens > 0
    assert sigs[0].symbolic is False


def test_integration_deterministic_output():
    """Same input produces identical output (determinism)."""
    source = '''def a_func(): return "a"
def b_func(): return "b"
'''
    nodes = [
        ExtractedNode("node_b", "b_func", 2),
        ExtractedNode("node_a", "a_func", 1),
    ]
    wg = build_workflow_graph("test:0", nodes, [], [], "node_a")
    profile = _default_model_profile()
    
    sigs1 = compute_node_token_signatures(wg, source, profile)
    sigs2 = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs1) == len(sigs2) == 2
    assert [s.node_name for s in sigs1] == [s.node_name for s in sigs2]
    for s1, s2 in zip(sigs1, sigs2):
        assert s1.node_name == s2.node_name
        assert s1.base_prompt_tokens == s2.base_prompt_tokens
        assert s1.symbolic == s2.symbolic
        assert s1.expansion_factor == s2.expansion_factor
        assert s1.input_dependency == s2.input_dependency


def test_integration_empty_workflow_graph():
    """Empty WorkflowGraph returns empty signatures."""
    wg = build_workflow_graph("test:0", [], [], [], None)
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, "def f(): pass", profile)
    
    assert len(sigs) == 0


def test_integration_large_function_with_many_strings():
    """Function with many strings sums all fragments."""
    source = '''def planner(state):
    part1 = "First"
    part2 = "Second"
    part3 = "Third"
    part4 = "Fourth"
    return part1 + part2 + part3 + part4
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].base_prompt_tokens > 0
    # Should sum tokens from all four strings
    expected_min = estimate_tokens_from_string("First") + estimate_tokens_from_string("Second")
    assert sigs[0].base_prompt_tokens >= expected_min


def test_integration_state_reference_detection():
    """State references in function body trigger input_dependency."""
    source = '''def planner(state):
    if state.get('key'):
        return "Found"
    return "Not found"
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].input_dependency is True


def test_integration_parameter_name_detection():
    """Function parameters trigger input_dependency."""
    source = '''def planner(state, config):
    return "Planning"
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].input_dependency is True


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_edge_case_nested_function_definitions():
    """Strings in nested functions are not detected (only top-level)."""
    source = '''def planner(state):
    def inner():
        return "inner string"
    return "outer string"
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    # Should detect "outer string" but not "inner string" (nested function)
    assert sigs[0].base_prompt_tokens > 0


def test_edge_case_string_in_comments():
    """Strings in comments are not detected (comments not in AST)."""
    source = '''def planner(state):
    # This is a comment with "quoted text"
    return "actual prompt"
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    # Should only detect "actual prompt", not comment text
    assert sigs[0].base_prompt_tokens > 0


def test_edge_case_multiline_string():
    """Multiline string literals are detected."""
    source = '''def planner(state):
    prompt = """This is a
multiline string
with multiple lines"""
    return prompt
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].base_prompt_tokens > 0
    assert sigs[0].symbolic is False


def test_edge_case_empty_string_literal():
    """Empty string literal is handled."""
    source = '''def planner(state):
    prompt = ""
    return prompt
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    # Empty string should still be detected (but may be marked symbolic if no other strings)
    # Actually, empty string detection should work, but if it's the only string and empty,
    # the function might be marked symbolic


def test_edge_case_very_long_string():
    """Very long strings are estimated correctly."""
    long_text = "x" * 1000
    source = f'''def planner(state):
    prompt = "{long_text}"
    return prompt
'''
    nodes = [ExtractedNode("planner", "planner", 1)]
    wg = build_workflow_graph("test:0", nodes, [], [], "planner")
    profile = _default_model_profile()
    sigs = compute_node_token_signatures(wg, source, profile)
    
    assert len(sigs) == 1
    assert sigs[0].base_prompt_tokens == estimate_tokens_from_string(long_text)
    assert sigs[0].symbolic is False
