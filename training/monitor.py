from torch.utils.tensorboard import SummaryWriter
import subprocess
import time
import signal
from ctypes import cdll
libc = cdll.LoadLibrary('libc.so.6')


class TensorboardMonitor:
    """
    provide a summary writer and a Tensorboard.dev uploader
    the uploader runs in the background
    using the default logdir of runs/**CURRENT_DATETIME_HOSTNAME**
    """

    def __init__(self):
        self.writer = SummaryWriter()
        self.uploader = subprocess.Popen(
            ["tensorboard", "dev", "upload", "--logdir", "runs"],
            preexec_fn=lambda *args: libc.prctl(1, signal.SIGTERM, 0, 0, 0),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        time.sleep(2)
        status = self.uploader.poll()
        if status is None:
            print("Tensorboard.dev uploader running ... ")
        else:
            print("Tensorboard.dev uploader exit with code ", status)

    def stop(self):
        self.uploader.terminate()
