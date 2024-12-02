"""Utility functions."""


def read_cli(parser):
    """Reads command line arguments."""

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file or JSON string",
    )
    parser.add_argument(
        "--json_config",
        action="store_true",
        help="Whether the config is a JSON string",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        required=False,
        default=None,
        help="Name of the run in wandb",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default="scOT",
        help="Name of the wandb project",
    )
    parser.add_argument(
        "--max_num_train_time_steps",
        type=int,
        default=None,
        help="Maximum number of time steps to use for training and validation.",
    )
    parser.add_argument(
        "--train_time_step_size",
        type=int,
        default=None,
        help="Time step size to use for training and validation.",
    )
    parser.add_argument(
        "--train_small_time_transition",
        action="store_true",
        help="Whether to train only for next step prediction.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Base path to data.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory. Will be prepended by wandb project and run name.",
    )
    parser.add_argument(
        "--disable_tqdm",
        action="store_true",
        help="Whether to disable tqdm progress bar",
    )
    parser.add_argument(
        "--push_to_hf_hub",
        type=str,
        default=None,
        help="Whether to push the model to Huggingface Hub. Specify the model repository name.",
    )
    parser.add_argument(
        "--just_velocities",
        action="store_true",
        help="Whether to only use velocities as input. Only relevant for incompressible flow datasets.",
    )
    parser.add_argument(
        "--move_data",
        type=str,
        default=None,
        help="If set, moves the data to this directory and trains from there.",
    )
    return parser


def get_num_parameters(model):
    """Returns the number of trainable parameters in a model."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_num_parameters_no_embed(model):
    """Returns the number of trainable parameters in a scOT model without embedding and recovery."""
    out = 0
    for name, p in model.named_parameters():
        if not ("embeddings" in name or "patch_recovery" in name) and p.requires_grad:
            out += p.numel()
    return out
