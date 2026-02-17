"""
Tests for model profile loader: YAML loading, dict construction, defaults, error handling.
"""

import tempfile
from pathlib import Path

import pytest

from noctyl.estimation import ModelProfile, default_model_profile, load_model_profile


def test_default_model_profile():
    """Returns correct default values."""
    profile = default_model_profile()
    assert profile.name == "default"
    assert profile.expansion_factor == 1.3
    assert profile.output_ratio == 0.5
    assert profile.pricing_input_per_1k == 0.0
    assert profile.pricing_output_per_1k == 0.0
    # Verify ModelProfile is frozen dataclass
    with pytest.raises(Exception):  # dataclass.FrozenInstanceError
        profile.name = "changed"


def test_load_model_profile_none():
    """None returns default."""
    profile = load_model_profile(None)
    assert profile == default_model_profile()


def test_load_model_profile_modelprofile():
    """ModelProfile passthrough."""
    original = ModelProfile("custom", 1.5, 0.6, 0.1, 0.2)
    result = load_model_profile(original)
    assert result is original
    assert result.name == "custom"
    assert result.expansion_factor == 1.5


def test_load_model_profile_dict():
    """Dict construction."""
    data = {
        "name": "test",
        "expansion_factor": 1.4,
        "output_ratio": 0.7,
        "pricing_input_per_1k": 0.01,
        "pricing_output_per_1k": 0.02,
    }
    profile = load_model_profile(data)
    assert profile.name == "test"
    assert profile.expansion_factor == 1.4
    assert profile.output_ratio == 0.7
    assert profile.pricing_input_per_1k == 0.01
    assert profile.pricing_output_per_1k == 0.02


def test_load_model_profile_dict_missing_pricing():
    """Dict with missing pricing fields uses defaults."""
    data = {
        "name": "test",
        "expansion_factor": 1.4,
        "output_ratio": 0.7,
    }
    profile = load_model_profile(data)
    assert profile.pricing_input_per_1k == 0.0
    assert profile.pricing_output_per_1k == 0.0


def test_load_model_profile_dict_nested_pricing():
    """Dict with nested pricing dict."""
    data = {
        "name": "test",
        "expansion_factor": 1.4,
        "output_ratio": 0.7,
        "pricing": {
            "input_per_1k": 0.01,
            "output_per_1k": 0.02,
        },
    }
    profile = load_model_profile(data)
    assert profile.pricing_input_per_1k == 0.01
    assert profile.pricing_output_per_1k == 0.02


def test_load_model_profile_yaml_file():
    """YAML file loading."""
    yaml_content = """
name: gpt-4o
expansion_factor: 1.2
output_ratio: 0.6
pricing:
  input_per_1k: 0.005
  output_per_1k: 0.015
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        # Load via str path
        profile1 = load_model_profile(yaml_path)
        assert profile1.name == "gpt-4o"
        assert profile1.expansion_factor == 1.2
        assert profile1.output_ratio == 0.6
        assert profile1.pricing_input_per_1k == 0.005
        assert profile1.pricing_output_per_1k == 0.015
        
        # Load via Path object
        profile2 = load_model_profile(Path(yaml_path))
        assert profile2.name == "gpt-4o"
    finally:
        Path(yaml_path).unlink()


def test_load_model_profile_yaml_multi_profile():
    """Multi-profile YAML file."""
    yaml_content = """
model_profiles:
  gpt-4o:
    expansion_factor: 1.2
    output_ratio: 0.6
    pricing:
      input_per_1k: 0.005
      output_per_1k: 0.015
  claude-3:
    expansion_factor: 1.1
    output_ratio: 0.5
    pricing:
      input_per_1k: 0.003
      output_per_1k: 0.015
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        # Should load first profile
        profile = load_model_profile(yaml_path)
        assert profile.name == "gpt-4o"
        assert profile.expansion_factor == 1.2
    finally:
        Path(yaml_path).unlink()


def test_load_model_profile_yaml_missing_fields():
    """Missing fields use defaults."""
    yaml_content = """
expansion_factor: 1.2
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        profile = load_model_profile(yaml_path)
        assert profile.expansion_factor == 1.2
        assert profile.output_ratio == 0.5  # default
        assert profile.pricing_input_per_1k == 0.0  # default
        assert profile.pricing_output_per_1k == 0.0  # default
    finally:
        Path(yaml_path).unlink()


def test_load_model_profile_yaml_invalid_file():
    """Invalid file path handling."""
    with pytest.raises(FileNotFoundError):
        load_model_profile("/nonexistent/path/profile.yaml")


def test_load_model_profile_yaml_invalid_syntax():
    """Invalid YAML syntax handling."""
    yaml_content = """
invalid: yaml: syntax: [error
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        with pytest.raises((ValueError, Exception)):  # yaml.YAMLError wrapped
            load_model_profile(yaml_path)
    finally:
        Path(yaml_path).unlink()


def test_load_model_profile_yaml_empty_file():
    """Empty YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = f.name
    
    try:
        with pytest.raises(ValueError, match="empty"):
            load_model_profile(yaml_path)
    finally:
        Path(yaml_path).unlink()


def test_load_model_profile_yaml_no_model_profiles_key():
    """YAML without model_profiles key (direct profile fields)."""
    yaml_content = """
name: direct-profile
expansion_factor: 1.3
output_ratio: 0.5
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        profile = load_model_profile(yaml_path)
        assert profile.name == "direct-profile"
        assert profile.expansion_factor == 1.3
    finally:
        Path(yaml_path).unlink()


def test_load_model_profile_deterministic():
    """Deterministic output."""
    data = {
        "name": "test",
        "expansion_factor": 1.4,
        "output_ratio": 0.7,
    }
    profile1 = load_model_profile(data)
    profile2 = load_model_profile(data)
    assert profile1 == profile2


def test_load_model_profile_dict_missing_expansion_factor():
    """Dict missing required expansion_factor."""
    data = {
        "name": "test",
        "output_ratio": 0.7,
    }
    with pytest.raises(ValueError, match="expansion_factor"):
        load_model_profile(data)


def test_load_model_profile_dict_invalid_type():
    """Dict with invalid type for expansion_factor."""
    data = {
        "name": "test",
        "expansion_factor": "not-a-number",
        "output_ratio": 0.7,
    }
    with pytest.raises(ValueError, match="number"):
        load_model_profile(data)


def test_load_model_profile_unsupported_type():
    """Unsupported source type."""
    with pytest.raises(TypeError):
        load_model_profile(123)  # int not supported


def test_profile_loader_error_handling_comprehensive():
    """Comprehensive error handling for all error types."""
    # FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_model_profile("/nonexistent/path/profile.yaml")
    
    # Invalid YAML syntax
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: syntax: [error\n")
        yaml_path = f.name
    
    try:
        with pytest.raises((ValueError, Exception)):
            load_model_profile(yaml_path)
    finally:
        Path(yaml_path).unlink()
    
    # Missing required field
    data_missing = {
        "name": "test",
        # Missing expansion_factor
    }
    with pytest.raises(ValueError, match="expansion_factor"):
        load_model_profile(data_missing)
    
    # Invalid type
    data_invalid = {
        "name": "test",
        "expansion_factor": "not-a-number",
    }
    with pytest.raises(ValueError, match="number"):
        load_model_profile(data_invalid)
    
    # Empty YAML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = f.name
    
    try:
        with pytest.raises(ValueError, match="empty"):
            load_model_profile(yaml_path)
    finally:
        Path(yaml_path).unlink()


def test_profile_loader_deterministic_permutations():
    """Same data, different construction order → identical profile."""
    # Dict with fields in different orders
    data1 = {
        "name": "test",
        "expansion_factor": 1.4,
        "output_ratio": 0.7,
        "pricing_input_per_1k": 0.01,
        "pricing_output_per_1k": 0.02,
    }
    data2 = {
        "pricing_output_per_1k": 0.02,
        "pricing_input_per_1k": 0.01,
        "output_ratio": 0.7,
        "expansion_factor": 1.4,
        "name": "test",
    }
    
    profile1 = load_model_profile(data1)
    profile2 = load_model_profile(data2)
    
    # Should be identical
    assert profile1 == profile2
    assert profile1.name == profile2.name
    assert profile1.expansion_factor == profile2.expansion_factor
    assert profile1.output_ratio == profile2.output_ratio


def test_profile_loader_large_yaml_file():
    """Multi-profile YAML with many profiles."""
    # Create YAML with 10 profiles
    profiles = {}
    for i in range(10):
        profiles[f"profile_{i}"] = {
            "expansion_factor": 1.0 + (i * 0.1),
            "output_ratio": 0.5 + (i * 0.05),
            "pricing": {
                "input_per_1k": 0.001 * i,
                "output_per_1k": 0.002 * i,
            },
        }
    
    yaml_content = "model_profiles:\n"
    for name, data in profiles.items():
        yaml_content += f"  {name}:\n"
        yaml_content += f"    expansion_factor: {data['expansion_factor']}\n"
        yaml_content += f"    output_ratio: {data['output_ratio']}\n"
        yaml_content += "    pricing:\n"
        yaml_content += f"      input_per_1k: {data['pricing']['input_per_1k']}\n"
        yaml_content += f"      output_per_1k: {data['pricing']['output_per_1k']}\n"
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        # Should load first profile
        profile = load_model_profile(yaml_path)
        assert profile.name == "profile_0"
        assert profile.expansion_factor == 1.0
        assert profile.output_ratio == 0.5
    finally:
        Path(yaml_path).unlink()
