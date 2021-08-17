import tempfile
from pathlib import Path
from heareval.tasks.runner import run
from heareval import tasks
from importlib import import_module
from mock import patch
from contextlib import ExitStack

from subprocess import Popen, PIPE
import threading
from multiprocessing import Process, Queue


class TestLuigiPipeline:
    def setup(self):
        self.test_dir = Path("./test_output")
        self.test_dir.mkdir(exist_ok=True)
        # self.tempdir = Path(
        #     "/Users/khumairraj/Desktop/audio/audionew/hear2021-eval-kit/test_output/tmp_zfb62uz"
        # )

    def teardown(self):
        del self.test_dir

    def _run_pipeline(self, task, dataset_fraction, suffix):
        # Make a temporary directory
        temp_dir = Path(tempfile.mkdtemp(dir=self.test_dir))
        # Get the config of the task and change the dataset fraction to test the dataset
        # fraction
        task_module = import_module(f"heareval.tasks.{task}")
        task_config = task_module.config
        task_config["small"]["dataset_fraction"] = 1
        task_config["small"]["version"] = "{suffix}_{version}".format(
            suffix = suffix, version=task_config["small"]["version"]
        )

        # Get the command line arguments to run the task
        args = (
            f"heareval.tasks.runner {task}"
            f" --small"
            f" --luigi-dir {temp_dir.joinpath('luigi_dir')}"
            f" --tasks-dir {temp_dir.joinpath('task_dir')}"
        )

        with ExitStack() as stack:
            stack.enter_context(patch(f"heareval.tasks.{task}.config", task_config))
            stack.enter_context(patch("sys.argv", args.split()))
            queue = Queue()
            p = Process(target=run())
            p.start()
            p.join() # this blocks until the process terminates
            result = queue.get()
            print(result)

        return temp_dir
    
    def test_run_pipeline_cli(self):
        p = Popen(["python", "-m", "heareval.tasks.runner", "speech_commands", "--small"], stdout=PIPE)
        stdout, _ = p.communicate()
        assert stdout == b"Hello World!\n"

    def test_speech_commands(self):
        temp_dir1 = self._run_pipeline(
            task="speech_commands", dataset_fraction=0.5, suffix="test1"
        )
        temp_dir2 = self._run_pipeline(
            task="speech_commands", dataset_fraction=0.5, suffix="test2"
        )
        all_in_temp1 = set(temp_dir1.glob("*"))
        all_int_temp2 = set(temp_dir2.glob("*"))
        print(all_in_temp1[:5])
        print(all_int_temp2[:5])
        assert all_in_temp1 == all_int_temp2


if __name__ == "__main__":
    task = "speech_commands"
    test = TestLuigiPipeline()
    test.setup()
    test.test_speech_commands()
