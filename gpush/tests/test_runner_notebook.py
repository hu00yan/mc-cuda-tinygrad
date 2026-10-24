"""The generated runner notebook, checked on the Mac before it costs a session.

The Kaggle path generates a notebook and pushes it. A notebook that does not
parse, or that leaves a substitution token behind, burns a GPU session on a
mistake that was visible offline. These tests are the reason `run` can refuse
before the push.

    Offline. No Kaggle quota, no network.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from gpush.backends.runner_nb import (build_runner_notebook,  # noqa: E402
                                      validate)
from gpush.manifest import build              # noqa: E402
from gpush.probes import DEVICE_PROBE         # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CMD = ["python", "derivation/divide.py", "mechanism_sentence"]


def _nb(cmd=None):
    m = build(REPO, None, include=["derivation"])
    m.compute_fingerprint("kaggle", "tenghuzhang/gpush-smoke-payload")
    return build_runner_notebook(m, "tenghuzhang/gpush-smoke-payload",
                                 cmd or CMD, "gpush-smoke"), m


class TestGeneratedNotebook(unittest.TestCase):
    def test_it_parses(self):
        nb, _ = _nb()
        self.assertEqual(validate(nb), [])

    def test_it_is_a_valid_notebook_envelope(self):
        nb, _ = _nb()
        self.assertEqual(nb["nbformat"], 4)
        self.assertGreaterEqual(nb["nbformat_minor"], 5)
        for i, c in enumerate(nb["cells"]):
            self.assertTrue(c.get("id"), f"cell {i} has no id")
            self.assertIn(c["cell_type"], ("code", "markdown"))
            self.assertTrue(isinstance(c["source"], list))

    def test_no_substitution_token_survives(self):
        """The templating is by sentinel, and _fill asserts none is left.

        The cells are real Python that itself uses % formatting and contains
        braces and a literal '$'. Every ordinary template language collides
        with that, and the collision is silent: you get a notebook that parses
        and does the wrong thing.
        """
        nb, _ = _nb()
        for i, c in enumerate(nb["cells"]):
            self.assertNotIn("@@", "".join(c["source"]),
                             f"cell {i} still holds a template token")

    def test_the_fingerprint_is_baked_into_the_verifying_cell(self):
        nb, m = _nb()
        src = "".join(nb["cells"][1]["source"])
        self.assertIn(m.fingerprint, src)
        self.assertIn("GPUSH_DEVICE_JSON", src)

    def test_it_verifies_the_tree_not_just_the_archive(self):
        """Verified against a real run: Kaggle mounts payload.tgz as a FILE.

        Nothing unpacks it. A runner that assumed an extracted directory found
        no tree root and refused to run -- correctly, and expensively.
        """
        nb, _ = _nb()
        src = "".join(nb["cells"][1]["source"])
        self.assertIn("extractall", src)
        self.assertIn("payload", src)

    def test_the_command_is_carried_verbatim(self):
        nb, _ = _nb(["python", "-c", "print('a b')", "--flag=x"])
        src = "".join(nb["cells"][2]["source"])
        self.assertIn('"-c"', src)
        self.assertIn("print('a b')", src)

    def test_metadata_records_the_provenance_fields(self):
        nb, m = _nb()
        g = nb["metadata"]["gpush"]
        self.assertEqual(g["fingerprint"], m.fingerprint)
        self.assertEqual(g["command"], CMD)
        self.assertIn("version", g)

    def test_the_probe_is_embedded_and_is_the_one_we_test(self):
        nb, _ = _nb()
        src = "".join(nb["cells"][1]["source"])
        self.assertIn(DEVICE_PROBE.splitlines()[1].strip(), src)

    def test_a_broken_cell_is_caught(self):
        """validate() must actually fail, or it proves nothing."""
        nb, _ = _nb()
        nb["cells"][2]["source"] = ["def broken(:\n", "    pass\n"]
        self.assertTrue(validate(nb))

    def test_a_missing_cell_id_is_caught(self):
        nb, _ = _nb()
        del nb["cells"][1]["id"]
        self.assertTrue(validate(nb))


if __name__ == "__main__":
    unittest.main()
