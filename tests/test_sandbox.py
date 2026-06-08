"""Tests for the sandbox — these back the isolation claims in the README, so
they use a REAL kernel (slower). State, error capture, read-only data, and the
timeout-drain fix are all exercised on a tiny throwaway CSV."""

import pytest

from da_verify.sandbox import KernelSandbox


@pytest.fixture
def csv(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    return p


def test_load_and_state_persists(csv):
    with KernelSandbox(data_csv=csv, cell_timeout=15) as sb:
        r = sb.execute("df = pd.read_csv(CSV_PATH); print(df.shape[0])")
        assert r.ok and r.stdout.strip() == "3"
        r2 = sb.execute("print(int(df['a'].sum()))")  # df survives across cells
        assert r2.ok and r2.stdout.strip() == "9"


def test_error_is_captured_not_raised(csv):
    with KernelSandbox(data_csv=csv, cell_timeout=15) as sb:
        r = sb.execute("1/0")
        assert not r.ok and "ZeroDivisionError" in r.error


def test_source_data_is_read_only(csv):
    with KernelSandbox(data_csv=csv, cell_timeout=15) as sb:
        r = sb.execute("open(CSV_PATH, 'a').write('x')")
        assert not r.ok and "Permission" in r.error


def test_timeout_interrupts_and_next_cell_is_clean(csv):
    # The regression we fixed in W2: post-timeout message bleed corrupting the
    # next cell's captured output.
    with KernelSandbox(data_csv=csv, cell_timeout=2) as sb:
        sb.execute("df = pd.read_csv(CSV_PATH)")
        r = sb.execute("import time; time.sleep(20)")
        assert r.timed_out
        r2 = sb.execute("print('alive', df.shape[0])")
        assert r2.ok and r2.stdout.strip() == "alive 3"  # not corrupted
