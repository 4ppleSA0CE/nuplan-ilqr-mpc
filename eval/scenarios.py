"""Load the committed scenario list's scenarios straight from the nuPlan mini split, outside the simulator."""
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml

import nuplan
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

REPO = Path(__file__).resolve().parents[1]
DEVKIT_CFG = Path(nuplan.__file__).parent / "planning/script/config/common"
MINI = Path(os.environ.get("NUPLAN_DATA_ROOT", "")) / "nuplan-v1.1/splits/mini"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def challenge_filter() -> dict:
    """The devkit's challenge-scenario filter as a dict (Hydra keys included)."""
    return _yaml(DEVKIT_CFG / "scenario_filter/nuplan_challenge_scenarios.yaml")


def listed() -> List[Dict[str, str]]:
    """Rows of eval/scenarios.csv."""
    with open(REPO / "eval/scenarios.csv") as f:
        return list(csv.DictReader(f))


def load(tokens: Optional[List[str]] = None, log_names: Optional[List[str]] = None) -> List[AbstractScenario]:
    """Challenge-type scenarios from mini, as the simulator builds them; optionally only these tokens and logs."""
    mapping = _yaml(DEVKIT_CFG / "scenario_builder/scenario_mapping/nuplan_scenario_mapping.yaml")
    builder = NuPlanScenarioBuilder(
        data_root=str(MINI),
        map_root=os.environ["NUPLAN_MAPS_ROOT"],
        sensor_root=os.environ["NUPLAN_DATA_ROOT"] + "/nuplan-v1.1/sensor_blobs",
        db_files=None,
        map_version="nuplan-maps-v1.0",
        scenario_mapping=ScenarioMapping(mapping["scenario_map"], mapping["subsample_ratio_override"]),
        verbose=False,
    )
    fields = {k: v for k, v in challenge_filter().items() if not k.startswith("_")}
    fields.update(scenario_tokens=tokens, log_names=log_names)
    return builder.get_scenarios(ScenarioFilter(**fields), Sequential())
