"""verl main_ppo entry point with dynamic sampling (scripts/dynamic_sampling.py).

Runs scripts/main_ppo_plugins.py unchanged, as __main__ (so Ray pickles its TaskRunner by value,
as when it is started directly), with its plugin file switched from scripts/adv_estimators.py to
scripts/dynamic_sampling.py, which loads scripts/adv_adaptive_threshold.py (and, through it,
adv_estimators.py) and scripts/adv_fixed_threshold.py: every custom estimator is available.
Dynamic sampling itself stays off unless +algorithm.dynamic_sampling=True. The switch is made here
rather than through FRONTIERSMITH_ADV_PLUGIN because scripts/launch_common.sh forwards only a fixed
list of variables into the container; TRAIN_ENTRY is one of them.

Usage: python scripts/main_ppo_dynamic.py <hydra overrides...>
(scripts/launch_dynamic_sampling_grpo.sh selects it with TRAIN_ENTRY=scripts/main_ppo_dynamic.py;
FRONTIERSMITH_PLUGIN_SELFTEST=1 runs main_ppo_plugins.py's Ray-actor self-test with this plugin)
"""

import os
import runpy

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    os.environ["FRONTIERSMITH_ADV_PLUGIN"] = os.path.join(SCRIPTS_DIR, "dynamic_sampling.py")
    runpy.run_path(os.path.join(SCRIPTS_DIR, "main_ppo_plugins.py"), run_name="__main__")
