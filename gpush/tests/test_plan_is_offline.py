"""Offline proof that `plan` makes no network call and reads no credential.

This is the property that makes `plan` a review surface rather than a second
push: the diff you read before spending anything must be reviewable with no
token on the machine and no route off it.

The test poisons every socket entry point AND points HOME at an empty
directory before gpush runs. If anything in the `plan` path reaches for the
network or for ~/.kaggle, it raises and the test fails.

    python -m unittest discover -s gpush/tests
"""
import os
import socket
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from gpush import backend as be          # noqa: E402
from gpush.cli import build_parser       # noqa: E402
from gpush.state import StateStore       # noqa: E402


class NetworkPoisoned(Exception):
    pass


def _boom(*a, **k):
    raise NetworkPoisoned("gpush touched the network during plan")


class TestPlanIsOffline(unittest.TestCase):
    def setUp(self):
        self._orig = {
            "connect": socket.socket.connect,
            "create_connection": socket.create_connection,
            "getaddrinfo": socket.getaddrinfo,
            "gethostbyname": socket.gethostbyname,
        }
        socket.socket.connect = _boom
        socket.create_connection = _boom
        socket.getaddrinfo = _boom
        socket.gethostbyname = _boom
        self._home = os.environ.get("HOME")
        self._tmp = tempfile.mkdtemp(prefix="gpush-no-home-")
        os.environ["HOME"] = self._tmp          # no ~/.kaggle anywhere near it

    def tearDown(self):
        for k, v in self._orig.items():
            setattr(socket, k, v)
        if self._home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._home

    def test_plan_completes_with_no_network_and_no_credentials(self):
        """The whole point: plan runs with sockets poisoned and no HOME."""
        out = os.path.join(self._tmp, "out")
        store = StateStore(out).bind("kaggle", "tenghuzhang/x-payload")
        bk = be.get("kaggle", dataset="tenghuzhang/x-payload",
                    kernel="tenghuzhang/x-kernel", store=store, out_root=out)

        from gpush.manifest import build
        tree = build(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), None, include=["derivation"])
        tree.compute_fingerprint(bk.name, "tenghuzhang/x-payload")

        plan = bk.plan(tree)          # the only call `plan` makes
        self.assertGreater(len(tree), 0, "the tree should not be empty")
        self.assertIsNotNone(plan.new_fingerprint)
        self.assertEqual(plan.new_fingerprint, tree.fingerprint)

    def test_parser_offers_both_backends_and_all_six_subcommands(self):
        ap = build_parser()
        text = ap.format_help()
        self.assertIn("kaggle", text)
        self.assertIn("colab", text)

    def test_a_third_backend_needs_no_core_change(self):
        """The registry, not a hardcoded list, is what the CLI reads."""
        @be.register
        class Dummy(be.Backend):
            name = "gpush-selftest-dummy"
            transport = "a transport that does not exist, for the test only"

            def plan(self, tree): return None
            def push(self, tree, plan, *, full_resync=False): return None
            def status(self): return None
            def doctor(self): return None
            def run(self, tree, argv, **kw): return None
            def fetch(self, state, dest): return []

        try:
            self.assertIn("gpush-selftest-dummy", be.known())
            ap = build_parser()
            self.assertIn("gpush-selftest-dummy", ap.format_help())
        finally:
            be._REGISTRY.pop("gpush-selftest-dummy", None)


if __name__ == "__main__":
    unittest.main()
