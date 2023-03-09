from argparse import ArgumentParser

from training.monitor import TensorboardMonitor


def init_settings() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "-n", "--name",
        type=str, help="name for upload to Tensorboard",
        default="Test")
    parser.add_argument(
        "-N", "--nsamples",
        type=int, help="Number of data samples",
        default=1e6)
    parser.add_argument(
        "-u", "--upload",
        type=bool, help="Upload to tensorboard.dev?",
        default=False)
    parser.add_argument(
        "-v", "--verbose", type=int, default=0, help="Verbosity level")
    args = parser.parse_args()

    global RUN_NAME, VERBOSE, TB_MONITOR_OBJ, TB_UPLOAD, N_SAMPLES
    RUN_NAME = args.name
    VERBOSE = args.verbose
    TB_UPLOAD = args.upload
    TB_MONITOR_OBJ = TensorboardMonitor(
        run_name=RUN_NAME, monitoring=TB_UPLOAD)
    N_SAMPLES = args.nsamples

    print("Settings:")
    print(f"    RUN_NAME  {RUN_NAME}")
    print(f"    TB_UPLOAD {TB_UPLOAD}")
    print(f"    N_SAMPLES {N_SAMPLES:,}")
    print(f"    VERBOSE   {VERBOSE}")
    return parser
