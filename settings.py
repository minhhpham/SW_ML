from argparse import ArgumentParser, BooleanOptionalAction

from training.monitor import TensorboardMonitor


def init_settings() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-e", "--exp_name",
        type=str,
        help="experiment name for upload to Tensorboard at exp_name/run_name",
        default="Test")
    parser.add_argument(
        "-r", "--run_name",
        type=str,
        help="run name for upload to Tensorboard at exp_name/run_name",
        default="Test0")
    parser.add_argument(
        "-N", "--nsamples",
        type=int, help="Number of data samples",
        default=1e6)
    parser.add_argument(
        "-m", "--monitor",
        type=bool, help="Monitor with Tensorboard?",
        default=False,
        action=BooleanOptionalAction
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=0, help="Verbosity level")
    args = parser.parse_args()

    global EXP_NAME, RUN_NAME, VERBOSE, TB_MONITOR, TB_MONITOR_OBJ, N_SAMPLES
    EXP_NAME = args.exp_name
    RUN_NAME = args.run_name
    VERBOSE = args.verbose
    TB_MONITOR = args.monitor
    TB_MONITOR_OBJ = TensorboardMonitor(
        monitoring=TB_MONITOR,
        exp_name=EXP_NAME,
        run_name=RUN_NAME,
    )
    N_SAMPLES = args.nsamples

    print("Settings:")
    print(f"    EXPERIMENT {EXP_NAME}")
    print(f"    RUN_NAME   {RUN_NAME}")
    print(f"    MONITORING {TB_MONITOR}")
    print(f"    N_SAMPLES  {N_SAMPLES:,}")
    print(f"    VERBOSE    {VERBOSE}")
    return parser
