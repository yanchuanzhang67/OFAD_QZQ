"""Open-loop integration: perception -> policy -> safety on an expert log.

Maps to the SDD Integration-Test tier:
  * open-loop ADE < 0.3 m vs expert trajectory
  * safety-filter coverage 100% over recorded hazard boundaries

These are skeletons: implement once the policy + replay modules exist.
"""
import pytest

pytestmark = pytest.mark.integration


def test_open_loop_ade_under_threshold():
    pytest.skip("requires expert log + trained IL policy")


def test_safety_filter_full_coverage():
    pytest.skip("requires recorded hazard-boundary dataset")
