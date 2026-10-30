"""verl main_ppo entry point that also loads this project's custom advantage estimators.

Identical to ``python -m verl.trainer.main_ppo`` except that scripts/adv_estimators.py
is imported inside the Ray TaskRunner actor (the process that computes advantages),
so names such as ``algorithm.adv_estimator=drgrpo_scaled`` resolve there.

Usage: python scripts/main_ppo_plugins.py <hydra overrides...>
(the training scripts select it with TRAIN_ENTRY=scripts/main_ppo_plugins.py)
"""

from __future__ import annotations

import os

import hydra
import ray

from verl.trainer.main_ppo import TaskRunner, run_ppo

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN_PATH = os.path.join(PROJECT_ROOT, "scripts", "adv_estimators.py")
VERL_CONFIG_DIR = os.path.join(PROJECT_ROOT, "verl", "verl", "trainer", "config")
PLUGIN_ENV = "FRONTIERSMITH_ADV_PLUGIN"  # set by main(); Ray workers inherit the driver's environment


def load_plugins() -> list[str]:
    """Import the plugin file by path (no package needed) and return the registered estimator names."""
    import importlib.util

    from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY

    path = os.environ[PLUGIN_ENV]
    spec = importlib.util.spec_from_file_location("frontiersmith_adv_estimators", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return sorted(ADV_ESTIMATOR_REGISTRY)


class PluginTaskRunner(TaskRunner):
    def check_plugins(self) -> list[str]:
        return load_plugins()

    def run(self, config):
        load_plugins()
        return super().run(config)


@hydra.main(config_path=VERL_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config):
    os.environ.setdefault(PLUGIN_ENV, PLUGIN_PATH)
    load_plugins()
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(PluginTaskRunner))


def _selftest() -> None:
    """FRONTIERSMITH_PLUGIN_SELFTEST=1: check the plugin loads inside a Ray actor (no GPUs, no training)."""
    os.environ.setdefault(PLUGIN_ENV, PLUGIN_PATH)
    ray.init(num_cpus=2, num_gpus=0, include_dashboard=False)
    runner = ray.remote(num_cpus=1)(PluginTaskRunner).remote()
    print("estimators registered inside the Ray actor:", ray.get(runner.check_plugins.remote()))


if __name__ == "__main__":
    if os.environ.get("FRONTIERSMITH_PLUGIN_SELFTEST") == "1":
        _selftest()
    else:
        main()
