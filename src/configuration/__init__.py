"""Strict assembly of module configs from the canonical system YAML."""

from configuration.system import SystemStackConfig, load_system_stack

__all__ = ["SystemStackConfig", "load_system_stack"]
