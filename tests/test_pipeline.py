import tempfile
from pathlib import Path
import shutil

from subprocess import Popen, PIPE


class TestLuigiPipeline:
    def setup(self):
        # Will change this path later. Currently for testing
        self.test_dir = Path("./test_output")
        self.test_dir.mkdir(exist_ok=True)

    def teardown(self):
        shutil.rmtree(self.test_dir)
        del self.test_dir

    def test_run_pipeline_cli(
        self, task="speech_commands", dataset_fraction=0.5, suffix="test1"
    ):
        """
        This function will call a subprocess with the luigi task

        Luigi Process have to run in a different process so that we can run multiple
        luigi pipeline in the same pytest and do appropriate comparisons
        """
        # Make a temporary directory
        temp_dir = Path(tempfile.mkdtemp(dir=self.test_dir))
        # Pass the arguments and run the luigi pipeline for these arguments
        # These arguments help configure the interior of the pipeline and thus can
        # mock arguments for the pipeline run
        args = [temp_dir, task, str(dataset_fraction), suffix]
        p = Popen(["python", "tests/luigi_subprocess.py"] + args, stdout=PIPE)
        p.communicate()
        return temp_dir

    def test_speech_commands(self):
        # Run the cli with for speech commands with the parameters
        test_dir_1 = self.test_run_pipeline_cli(
            task="speech_commands", dataset_fraction=0.5, suffix="test1"
        )
        # Rerun the cli with a different base folder to check the sampling stability
        test_dir_2 = self.test_run_pipeline_cli(
            task="speech_commands", dataset_fraction=0.5, suffix="test2"
        )
        # Get the files in test dir 1
        test_dir_1_files = list(
            map(lambda file: file.relative_to(test_dir_1), test_dir_1.glob("*"))
        )
        # Get the files in test dir 2
        test_dir_2_files = list(
            map(lambda file: file.relative_to(test_dir_2), test_dir_2.glob("*"))
        )
        # Check if they are same
        assert set(test_dir_1_files) == set(test_dir_2_files)


if __name__ == "__main__":
    # Run the pytest quickly for development. Remove later
    test = TestLuigiPipeline()
    test.setup()
    test.test_speech_commands()
