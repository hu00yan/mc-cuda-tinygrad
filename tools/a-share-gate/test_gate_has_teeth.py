"""门禁负控制：这一条**永远失败**，故意的。

它不是测试，是探针。``scripts/verify_gate.py`` 把 ``scripts/gate.sh`` 里那条
**真实的** fast 层 pytest 命令取出来，只跑这一个文件，要求退出码非零。

为什么需要它：fast 层目前的唯一自证是「slow 被排除了」，那证明的是*筛掉了什么*，
不证明*留下来的东西真的会被判红*。``run`` 里加一个 ``|| true``、把 ``if ! "$@"``
写成 ``if "$@"``、把 pytest 换成 ``true``——三种改动都让门禁永远绿，而输出形状
和全绿一模一样。绿看起来很健康，所以要用一条必然失败的测试把「会红」这件事钉住。

它默认**不进套件**：同目录的 ``conftest.py`` 用 ``collect_ignore`` 把它挡在
``pytest tests/`` 之外。放进默认套件 = 门禁永久变红 = 没人再看门禁 = 等于没有门禁。
"""

from __future__ import annotations

import pytest


def test_canary_deliberate_failure_the_gate_must_catch() -> None:
    """永远失败。门禁抓不到它，就说明门禁没有牙齿。"""
    pytest.fail(
        "canary（故意失败）：若你在正常套件里看到这条，说明 collect_ignore 失效了；"
        "若它在这里却让门禁变绿，说明 scripts/gate.sh 不再把失败当失败。"
    )
