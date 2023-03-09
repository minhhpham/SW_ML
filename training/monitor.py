import signal
import subprocess
import time
from ctypes import cdll

from torch.utils.tensorboard import SummaryWriter

import settings

libc = cdll.LoadLibrary('libc.so.6')


class TensorboardMonitor:
    """
    provide a summary writer and a Tensorboard.dev uploader
    the uploader runs in the background
    using the default logdir of runs/**CURRENT_DATETIME_HOSTNAME**
    """

    def __init__(self, background_upload=True):
        self.writer = SummaryWriter()
        if background_upload:
            self.uploader = subprocess.Popen(
                ["tensorboard", "dev", "upload",
                 "--logdir", "runs",
                 "--name", settings.RUN_NAME],
                preexec_fn=lambda *args: libc.prctl(
                    1, signal.SIGTERM, 0, 0, 0
                ),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            time.sleep(2)
            status = self.uploader.poll()
            if status is None:
                print("Tensorboard.dev uploader running ... ")
            else:
                print("Tensorboard.dev uploader exit with code ", status)
        else:
            self.uploader = None

    def stop(self):
        if self.uploader is not None:
            self.uploader.terminate()
