"""Layer 1: perception & multi-modal BEV / Occupancy fusion.

Import the heavy (torch-backed) modules explicitly so that the package itself
stays importable without torch installed::

    from perception.bev_fusion import BEVFusion

Internal boundaries are ``encoders`` (per-sensor representation), ``fusion``
(explicit modality-aware BEV fusion), and ``bev_fusion`` (public facade,
geometry and heads).
"""
