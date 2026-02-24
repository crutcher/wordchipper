from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "local_crates" / "wordchipper-bench" / "benches" / "data"


@pytest.fixture(scope="session")
def english_text():
    return (DATA_DIR / "english.txt").read_text() * 10


@pytest.fixture(scope="session")
def diverse_text():
    return (DATA_DIR / "multilingual.txt").read_text() * 10


@pytest.fixture(scope="session")
def english_lines():
    text = (DATA_DIR / "english.txt").read_text()
    return [line for line in text.splitlines() if line.strip()]


@pytest.fixture(scope="session")
def diverse_lines():
    text = (DATA_DIR / "multilingual.txt").read_text()
    return [line for line in text.splitlines() if line.strip()]


def pytest_terminal_summary(terminalreporter, config):
    try:
        benchmarks = config._benchmarksession.benchmarks
    except AttributeError:
        return
    if not benchmarks:
        return

    rows = []
    for bench in benchmarks:
        input_bytes = bench.extra_info.get("input_bytes")
        if not input_bytes or not bench.stats:
            continue
        mb_s = input_bytes / bench.stats.mean / 1_000_000
        rows.append((bench.group or "", bench.name, mb_s))

    if not rows:
        return

    terminalreporter.section("throughput", sep="-")
    current_group = None
    for group, name, mb_s in sorted(rows):
        if group != current_group:
            current_group = group
            terminalreporter.write_line(f"\n  {group}:")
        terminalreporter.write_line(f"    {name:55s} {mb_s:>10,.1f} MB/s")
