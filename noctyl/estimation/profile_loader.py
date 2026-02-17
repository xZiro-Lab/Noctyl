"""
Model profile loader: supports YAML files, dicts, ModelProfile instances, and defaults.
"""

from __future__ import annotations

from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from noctyl.estimation.data_model import ModelProfile


def default_model_profile() -> ModelProfile:
    """
    Return the default model profile.
    
    Returns:
        ModelProfile with name="default", expansion_factor=1.3, output_ratio=0.5,
        pricing fields set to 0.0.
    """
    return ModelProfile(
        name="default",
        expansion_factor=1.3,
        output_ratio=0.5,
        pricing_input_per_1k=0.0,
        pricing_output_per_1k=0.0,
    )


def load_model_profile(
    source: ModelProfile | str | Path | dict | None,
) -> ModelProfile:
    """
    Load a ModelProfile from various sources.
    
    Args:
        source: Can be:
            - ModelProfile instance: returned as-is
            - str or Path: treated as YAML file path
            - dict: constructed directly from dict keys
            - None: returns default_model_profile()
    
    Returns:
        ModelProfile instance
    
    Raises:
        FileNotFoundError: If source is a file path that doesn't exist
        yaml.YAMLError: If source is a YAML file with invalid syntax
        ValueError: If dict source is missing required fields or has invalid types
    """
    if source is None:
        return default_model_profile()
    
    if isinstance(source, ModelProfile):
        return source
    
    if isinstance(source, (str, Path)):
        return _load_from_yaml_file(source)
    
    if isinstance(source, dict):
        return _load_from_dict(source)
    
    raise TypeError(
        f"Unsupported source type for load_model_profile: {type(source).__name__}"
    )


def _load_from_yaml_file(path: str | Path) -> ModelProfile:
    """Load ModelProfile from a YAML file."""
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load profiles from YAML files. "
            "Install it with: pip install pyyaml"
        )
    
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Profile file not found: {file_path}")
    
    try:
        content = file_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise ValueError(f"Failed to parse YAML file {file_path}: {e}") from e
    
    if data is None:
        raise ValueError(f"YAML file {file_path} is empty")
    
    # Handle both formats:
    # 1. Direct profile fields at root
    # 2. Nested under model_profiles dict
    if isinstance(data, dict):
        if "model_profiles" in data:
            # Multi-profile format: extract first profile
            profiles = data["model_profiles"]
            if not isinstance(profiles, dict) or not profiles:
                raise ValueError(
                    f"YAML file {file_path}: 'model_profiles' must be a non-empty dict"
                )
            # Use first profile
            profile_name = next(iter(profiles.keys()))
            profile_data = profiles[profile_name]
            if not isinstance(profile_data, dict):
                raise ValueError(
                    f"YAML file {file_path}: profile '{profile_name}' must be a dict"
                )
            return _load_from_dict(profile_data, default_name=profile_name)
        else:
            # Single profile format: use root dict
            return _load_from_dict(data)
    
    raise ValueError(f"YAML file {file_path}: expected dict, got {type(data).__name__}")


def _load_from_dict(data: dict, default_name: str | None = None) -> ModelProfile:
    """
    Construct ModelProfile from a dict.
    
    Args:
        data: Dict with profile fields
        default_name: Default name if 'name' key is missing
    
    Returns:
        ModelProfile instance
    
    Raises:
        ValueError: If required fields are missing or have invalid types
    """
    # Extract name
    name = data.get("name", default_name or "default")
    if not isinstance(name, str):
        raise ValueError(f"Profile 'name' must be a string, got {type(name).__name__}")
    
    # Extract expansion_factor
    expansion_factor = data.get("expansion_factor")
    if expansion_factor is None:
        raise ValueError("Profile 'expansion_factor' is required")
    if not isinstance(expansion_factor, (int, float)):
        raise ValueError(
            f"Profile 'expansion_factor' must be a number, got {type(expansion_factor).__name__}"
        )
    expansion_factor = float(expansion_factor)
    
    # Extract output_ratio
    output_ratio = data.get("output_ratio", 0.5)
    if not isinstance(output_ratio, (int, float)):
        raise ValueError(
            f"Profile 'output_ratio' must be a number, got {type(output_ratio).__name__}"
        )
    output_ratio = float(output_ratio)
    
    # Extract pricing (can be nested or flat)
    pricing_input_per_1k = 0.0
    pricing_output_per_1k = 0.0
    
    if "pricing" in data:
        pricing = data["pricing"]
        if isinstance(pricing, dict):
            pricing_input_per_1k = pricing.get("input_per_1k", 0.0)
            pricing_output_per_1k = pricing.get("output_per_1k", 0.0)
        else:
            raise ValueError(
                f"Profile 'pricing' must be a dict, got {type(pricing).__name__}"
            )
    else:
        # Check for flat pricing fields
        pricing_input_per_1k = data.get("pricing_input_per_1k", 0.0)
        pricing_output_per_1k = data.get("pricing_output_per_1k", 0.0)
    
    # Validate pricing fields are numbers
    if not isinstance(pricing_input_per_1k, (int, float)):
        raise ValueError(
            f"Profile 'pricing_input_per_1k' must be a number, "
            f"got {type(pricing_input_per_1k).__name__}"
        )
    if not isinstance(pricing_output_per_1k, (int, float)):
        raise ValueError(
            f"Profile 'pricing_output_per_1k' must be a number, "
            f"got {type(pricing_output_per_1k).__name__}"
        )
    
    return ModelProfile(
        name=name,
        expansion_factor=float(expansion_factor),
        output_ratio=float(output_ratio),
        pricing_input_per_1k=float(pricing_input_per_1k),
        pricing_output_per_1k=float(pricing_output_per_1k),
    )
