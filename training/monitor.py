from torch.utils.tensorboard import SummaryWriter
import subprocess
import time


class TensorboardMonitor:
    """
    provide a summary writer and a Tensorboard.dev uploader
    the uploader runs in the background
    using the default logdir of runs/**CURRENT_DATETIME_HOSTNAME**
    """

    def __init__(self):
        self.writer = SummaryWriter()
        self.uploader = subprocess.Popen([
            "tensorboard", "dev", "upload", "--logdir", "runs"
        ])
        time.sleep(2)
        status = self.uploader.poll()
        if status is None:
            print("Tensorboard.dev uploader running ... ")
        else:
            print("Tensorboard.dev uploader exit with code ", status)

    def stop(self):
        self.uploader.terminate()
