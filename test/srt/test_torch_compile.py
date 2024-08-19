import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_child_process
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, popen_launch_server


class TestAccuracy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = "http://127.0.0.1:8157"
        cls.process = popen_launch_server(
            cls.model, cls.base_url, timeout=300, other_args=["--enable-torch-compile"]
        )

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=20,
            num_threads=20,
        )

        metrics = run_eval(args)
        assert metrics["score"] >= 0.5


if __name__ == "__main__":
    unittest.main(warnings="ignore")

    # t = TestAccuracy()
    # t.setUpClass()
    # t.test_mmlu()
    # t.tearDownClass()
