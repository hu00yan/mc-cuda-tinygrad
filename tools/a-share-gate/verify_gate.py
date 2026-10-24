#!/usr/bin/env python3
"""证明门禁**能红**——用一条必然失败的 canary 去撞它。

用法::

    uv run python scripts/verify_gate.py

**为什么需要这个脚本。** fast 层目前唯一的自证是「slow 层被排除了」
（`gate.sh` 会列出 1451 vs 1450）。那证明的是*筛掉了什么*，**不证明留下来的
东西真的会被判红**。而下面三种改动都让门禁永远绿，输出形状却和全绿一模一样：

- `run()` 里 `if ! "$@"` 写成 `if "$@"`；
- `run()` 的条件后面多一个 `|| true`；
- 那条 pytest 命令被换成 `true`（或 `-m "not slow"` 被换成 `-m "slow"`）。

绿看起来很健康，所以「会红」这件事必须自己有一条探针，否则门禁的失效是静默的。
这条探针在 `tests/canary/test_gate_has_teeth.py`，它**永远失败**，且默认不进套件
（同目录 `conftest.py` 的 `collect_ignore`）。

**它怎么做到「窄化」而仍然是同一条门禁。** 脚本不重写一条 pytest 命令，而是
**从 `scripts/gate.sh` 里把原文抠出来**：

1. `run()` 函数体逐字提取 —— 失败判定逻辑（`if ! "$@"` / `fail=1`）用的是真身，
   不是副本。所以「`run` 吞掉失败」这类改动会被抓到。
2. fast 层那条 pytest 命令行逐字提取，只把目标 `tests/` 换成 canary 的路径。
   所以「pytest 命令本身被改坏」也会被抓到。
3. 拼成一个只跑一条测试的驱动脚本，交给 bash 执行，要求它报告 `fail=1`。

抠不到那两段就直接报错退出——**形状变了不能算通过**。这是本脚本最重要的一条：
找不到要检查的东西时，通过是最危险的答案。

**它不证明的事**（写下来免得被当成更强的保证）：它不跑另外 5 道门（ruff / mypy /
verify / CAS 链接审计 / 站点同步）。那些门会不会红由它们自己的实现决定；这里只钉
「pytest 这一层的失败判定有没有牙齿」。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GATE = REPOSITORY_ROOT / "scripts" / "gate.sh"

#: 必然失败的那条负控制。必须与文件里的函数名一致，否则断言会「通过」得很轻松。
CANARY_PATH = "tests/canary/test_gate_has_teeth.py"
CANARY_NODE = "test_canary_deliberate_failure_the_gate_must_catch"

_RUN_DEF = re.compile(r"^run\(\) \{\n(.*?)^\}$", re.MULTILINE | re.DOTALL)


class GateShapeError(RuntimeError):
    """`gate.sh` 的形状变了，抠不出要检查的东西。"""


def _read_gate() -> str:
    if not GATE.is_file():
        raise GateShapeError(f"找不到门禁脚本：{GATE}")
    return GATE.read_text(encoding="utf-8")


def _run_function(text: str) -> str:
    """逐字取出 `run()` 函数体（含 `run() {` 与收尾的 `}`）。

    逐字而不是重写：这个函数的全部价值就在失败判定那三行
    （`if ! "$@"` / `fail=1` / 收尾），重写一份等于测我自己写的副本。
    """

    match = _RUN_DEF.search(text)
    if match is None:
        raise GateShapeError("gate.sh 里找不到 `run() { ... }` 定义")
    return f"run() {{\n{match.group(1)}}}\n"


def _fast_pytest_line(text: str) -> str:
    """取出 fast 层那条 `run "pytest 测试（fast）" ...` 命令行的原文。"""

    candidates = [
        line
        for line in text.splitlines()
        if line.lstrip().startswith("run ")
        and "python -m pytest tests/" in line
        and "not slow" in line
        and "--collect-only" not in line
    ]
    if len(candidates) != 1:
        raise GateShapeError(
            "gate.sh 里 fast 层 pytest 命令行应恰好 1 条，实际找到 "
            f"{len(candidates)} 条：{candidates}。门禁结构变了就更新本脚本，"
            "不要让它静默地什么都不检查。"
        )
    return candidates[0]


def _narrow(line: str) -> str:
    """把命令行里的目标 `tests/` 换成 canary 的路径，其余一字不改。"""

    if line.count("pytest tests/") != 1:
        raise GateShapeError(f"这条命令里 `pytest tests/` 不是恰好一处：{line}")
    narrowed = line.replace("pytest tests/", f"pytest {CANARY_PATH}")
    if CANARY_PATH not in narrowed:
        raise GateShapeError(f"窄化失败，canary 路径没进命令：{narrowed}")
    return narrowed


def _driver(text: str) -> str:
    """拼出只跑 canary 的驱动脚本。

    末尾不是 `exit $?`：`run()` 无论成败都返回 0（它内部 `tail | sed` 收尾），
    成败只体现在 `fail` 变量上。所以驱动必须读 `fail`，这正是真实门禁的判据。
    """

    return "\n".join(
        [
            "set -uo pipefail",
            "fail=0",
            _run_function(text),
            _narrow(_fast_pytest_line(text)),
            'if [ "$fail" -eq 0 ]; then echo "GATE_WOULD_PASS"; exit 0; fi',
            'echo "GATE_WOULD_FAIL"; exit 1',
        ]
    )


def _bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _collect_command(text: str) -> str:
    """同一条命令、同样的参数，只把「跑」换成「收集」，用于确认 canary 不在默认套件里。"""

    line = _fast_pytest_line(text)
    # 去掉 `run` 与它那个带空格的中文名字参数（名字里含空格，不能用 split 切）。
    match = re.match(r'^\s*run\s+"[^"]*"\s+(.*)$', line)
    if match is None:
        raise GateShapeError(f"这条 `run` 行没有可解析的名字参数：{line}")
    command = match.group(1).strip()
    if " -q" not in command:
        raise GateShapeError(f"命令里没有 `-q`，无法安全插入 `--collect-only`：{command}")
    return command.replace(" -q", " -q --collect-only", 1)


def main() -> int:
    try:
        text = _read_gate()
        driver = _driver(text)
        collect = _collect_command(text)
    except GateShapeError as error:
        print(f"gate has teeth: ERROR —— {error}", file=sys.stderr)
        return 1

    # 1) 默认套件里必须没有它。有 = 门禁永久变红 = 等于没有门禁。
    collected = _bash(collect)
    if collected.returncode != 0:
        print(
            "gate has teeth: ERROR —— 默认 fast 层的收集本身失败了，"
            f"无法判断 canary 是否被收进来：\n{collected.stdout[-2000:]}\n{collected.stderr[-2000:]}",
            file=sys.stderr,
        )
        return 1
    if CANARY_NODE in collected.stdout:
        print(
            "gate has teeth: ERROR —— canary 出现在默认套件的收集结果里，"
            "门禁会永久变红。检查 tests/canary/conftest.py 的 collect_ignore。",
            file=sys.stderr,
        )
        return 1

    # 2) 只跑 canary，要求门禁自己报告失败。
    ran = _bash(driver)
    output = f"{ran.stdout}\n{ran.stderr}"
    if ran.returncode == 0 or "GATE_WOULD_FAIL" not in output:
        print(
            "gate has teeth: FAIL —— 门禁把一条必然失败的测试放过去了。\n"
            "  这说明 scripts/gate.sh 的失败判定已经失效（`run()` 吞掉了失败，"
            "或者那条 pytest 命令被改坏了），所有绿灯都不可信。\n"
            f"  退出码：{ran.returncode}\n  输出：\n{output[-3000:]}",
            file=sys.stderr,
        )
        return 1
    if CANARY_NODE not in output or "failed" not in output:
        print(
            "gate has teeth: FAIL —— 门禁红了，但不是因为 canary 那条测试失败"
            "（例如收集失败、路径写错）。非零退出码不等于「抓到了失败」。\n"
            f"  输出：\n{output[-3000:]}",
            file=sys.stderr,
        )
        return 1

    print("gate has teeth: OK")
    print(f"  · 默认 fast 层不收集 canary（{CANARY_PATH}）")
    print("  · 把 canary 单独交给 gate.sh 的 run()+pytest 命令后，门禁判红")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
