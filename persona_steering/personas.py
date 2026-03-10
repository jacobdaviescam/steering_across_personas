"""Persona loading from YAML configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml

from persona_steering.config import PERSONAS_DIR, PersonaConfig


def load_persona(name: str, personas_dir: Path = PERSONAS_DIR) -> PersonaConfig:
    """Load a persona config from YAML."""
    path = personas_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No persona config at {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return PersonaConfig(
        name=data["name"],
        slug=data.get("slug", name),
        system_prompt_variants=data.get("system_prompt_variants", []),
        few_shot_examples=data.get("few_shot_examples", []),
        activation_injection=data.get("activation_injection"),
        description=data.get("description", ""),
        tags=data.get("tags", []),
    )


def load_all_personas(personas_dir: Path = PERSONAS_DIR) -> list[PersonaConfig]:
    """Load all persona configs from a directory."""
    configs = []
    for path in sorted(personas_dir.glob("*.yaml")):
        configs.append(load_persona(path.stem, personas_dir))
    return configs
