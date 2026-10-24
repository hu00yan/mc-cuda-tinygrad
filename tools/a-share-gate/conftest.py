"""canary 不进默认套件。

``tests/canary/test_gate_has_teeth.py`` 是一条**必然失败**的负控制，只有
``scripts/verify_gate.py`` 会显式指名它。默认套件把它收进来就会让门禁永久变红，
而永久红的门禁等于没有门禁。

``collect_ignore`` 只影响**目录递归收集**；显式传入该路径仍然会被收集——这正是
verify 脚本需要的语义（默认跑不到，指名跑得到）。这条语义由
``scripts/verify_gate.py`` 的第二个断言守着：它会跑一次默认 fast 层的
``--collect-only``，若 canary 出现在里面就报错。
"""

from __future__ import annotations

collect_ignore = ["test_gate_has_teeth.py"]
