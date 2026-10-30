"""verl main_ppo entry point for algorithm.adv_estimator=grpo_adaptive_threshold.

Runs scripts/main_ppo_plugins.py unchanged, as __main__ (so Ray pickles its TaskRunner by value,
as when it is started directly), with its plugin file switched from scripts/adv_estimators.py to
scripts/adv_adaptive_threshold.py (which loads adv_estimators.py too). The switch is made here
rather than through FRONTIERSMITH_ADV_PLUGIN because scripts/launch_common.sh forwards only a
fixed list of variables into the container; TRAIN_ENTRY is one of them.

Usage: python scripts/main_ppo_adaptive.py <hydra overrides...>
(scripts/launch_adaptive_threshold_grpo.sh selects it with TRAIN_ENTRY=scripts/main_ppo_adaptive.py;
FRONTIERSMITH_PLUGIN_SELFTEST=1 runs main_ppo_plugins.py's Ray-actor self-test with this plugin)
"""

import os
import runpy

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    os.environ["FRONTIERSMITH_ADV_PLUGIN"] = os.path.join(SCRIPTS_DIR, "adv_adaptive_threshold.py")
    runpy.run_path(os.path.join(SCRIPTS_DIR, "main_ppo_plugins.py"), run_name="__main__")
