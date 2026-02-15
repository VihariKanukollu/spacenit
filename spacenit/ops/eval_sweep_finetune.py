"""Launch fine-tune evaluation sweeps for SpaceNit and other models.

Usage examples:

1. Finetune all eval tasks using a baseline model (default lr only):
   python -m spacenit.ops.eval_sweep_finetune \\
       --project_name test_finetune \\
       --module_path path/to/launch_script.py \\
       --cluster your-org/your-cluster \\
       --model terramind \\
       --defaults_only

2. Finetune all eval tasks using a SpaceNit model (default lr only):
   python -m spacenit.ops.eval_sweep_finetune \\
       --checkpoint_path /path/to/checkpoint \\
       --project_name test_finetune \\
       --module_path scripts/official/base.py \\
       --cluster your-org/your-cluster \\
       --defaults_only

Flags:
  --defaults_only  Runs just one job: lr = 1e-4
  (omit)           Sweeps lrs: [1e-4, 5e-4, 1e-3]
"""

import argparse
import os
import subprocess  # nosec
from logging import getLogger

from spacenit.ops.constants import EVAL_WANDB_PROJECT

logger = getLogger(__name__)

# Learning rates to sweep over.
FT_LRS = [1e-4, 5e-4, 1e-3]


def build_commands(
    args: argparse.Namespace,
    extra_cli: list[str],
) -> list[str]:
    """Build the commands for the fine-tune evaluation sweep.

    Constructs a list of shell commands that launch fine-tune evaluation jobs
    across a grid of learning rates for the specified checkpoint or model preset.

    Args:
        args: parsed command-line arguments.
        extra_cli: additional CLI overrides to forward.

    Returns:
        list of shell command strings to execute.
    """
    project_name = args.project_name or EVAL_WANDB_PROJECT
    commands: list[str] = []
    # Placeholder: full finetune sweep logic to be implemented
    logger.info(f"Building finetune eval sweep commands for project {project_name}")
    return commands


def main() -> None:
    """Run the fine-tune evaluation sweep."""
    parser = argparse.ArgumentParser(description="Run finetune evaluation sweeps.")
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
        "--defaults_only", action="store_true", help="Only run with default lr"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print commands without launching"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model preset key"
    )
    parser.add_argument(
        "--use_dataset_normalizer",
        action="store_true",
        help="Use dataset statistics instead of pretrained normalizer",
    )
    parser.add_argument(
        "--finetune_seed", type=int, default=None, help="Base random seed"
    )

    args, extra_cli = parser.parse_known_args()

    env = os.environ.copy()
    env["FINETUNE"] = "1"
    commands_to_run = build_commands(args, extra_cli)
    for cmd in commands_to_run:
        logger.info(cmd)
        subprocess.run(cmd, shell=True, check=True, env=env)  # nosec


if __name__ == "__main__":
    main()
