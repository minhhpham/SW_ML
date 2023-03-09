import signal
import subprocess
import time
from ctypes import cdll

from torch.utils.tensorboard import SummaryWriter

libc = cdll.LoadLibrary('libc.so.6')


class DummySummaryWriter:
    """A dummy summary writer that does nothing
    """

    def add_scalar(self, *args, **kwargs):
        pass

    def add_graph(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass


class TensorboardMonitor:
    """
    provide a summary writer and a Tensorboard.dev uploader
    the uploader runs in the background
    using the default logdir of runs/**CURRENT_DATETIME_HOSTNAME**
    """

    def __init__(self, run_name: str, monitoring=True):
        if monitoring:
            self.writer = SummaryWriter()
            self.uploader = subprocess.Popen(
                ["tensorboard", "dev", "upload",
                 "--logdir", "runs",
                 "--name", run_name],
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
            self.writer = DummySummaryWriter()
            self.uploader = None

    def stop(self):
        if self.uploader is not None:
            self.uploader.terminate()
