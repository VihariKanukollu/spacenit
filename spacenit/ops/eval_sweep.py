"""Run an evaluation sweep for an arbitrary SpaceNit checkpoint.

Usage example:
    python -m spacenit.ops.eval_sweep \\
        --cluster=your-org/your-cluster \\
        --checkpoint_path=/path/to/checkpoint \\
        --module_path=scripts/my_experiment.py
"""

import argparse
import subprocess  # nosec
from logging import getLogger

from spacenit.ops.constants import EVAL_WANDB_PROJECT
from spacenit.ops.experiment import SubCmd

logger = getLogger(__name__)

LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
LAUNCH_OVERRIDES = "--launch.priority=high --launch.num_gpus=1 --launch.task_name=eval"


def build_commands(args: argparse.Namespace, extra_cli: list[str]) -> list[str]:
    """Build the commands for the evaluation sweep.

    Constructs a list of shell commands that launch evaluation jobs across
    a grid of hyperparameters (learning rates, normalization modes, pooling
    types) for the specified checkpoint or baseline model.

    Args:
        args: parsed command-line arguments.
        extra_cli: additional CLI overrides to forward.

    Returns:
        list of shell command strings to execute.
    """
    project_name = args.project_name or EVAL_WANDB_PROJECT
    commands_to_run: list[str] = []
    # Placeholder: full sweep logic to be implemented
    logger.info(f"Building eval sweep commands for project {project_name}")
    return commands_to_run


def main() -> None:
    """Run the full evaluation sweep or just the defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, required=True, help="Cluster name")
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Checkpoint path"
    )
    parser.add_argument(
        "--module_path", type=str, default=None, help="Path to module .py"
    )
    parser.add_argument(
        "--project_name", type=str, required=False, help="Wandb project name"
    )
    parser.add_argument(
        "--defaults_only", action="store_true", help="Only run with default values"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print commands without running"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Baseline model to use"
    )
    parser.add_argument(
        "--lr_only", action="store_true", help="Only sweep learning rates"
    )

    args, extra_cli = parser.parse_known_args()
    commands_to_run = build_commands(args, extra_cli)

    logger.info(f"Running {len(commands_to_run)} commands")
    for cmd in commands_to_run:
        logger.info(cmd)
        subprocess.run(cmd, shell=True, check=True)  # nosec
    logger.info(f"Finished running {len(commands_to_run)} commands")


if __name__ == "__main__":
    main()
