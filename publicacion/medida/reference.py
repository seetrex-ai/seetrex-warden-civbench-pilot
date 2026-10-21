# SPDX-License-Identifier: MIT
"""CivBench reference games: the 6 admissible Opus 4.6 games of the Ground Control scenario.

Reads the two parquet tables of the public dataset `civbench/civbench-v1` (games.parquet and
tool_calls.parquet, in one directory given by argument). Needs `pyarrow`; nothing else outside
the standard library.

As a module it gives `load_games(dir)`; as a script it prints the uncut get_victory_progress
rate of each game, the pooled rate and the per-game maximum:

    python reference.py TABLES_DIR
"""
import sys
from fractions import Fraction
from pathlib import Path

import pyarrow.parquet as pq

GVP = "get_victory_progress"
MODEL = "claude-opus-4-6"
SCENARIO = "ground_control"
N_GAMES = 6


class ReferenceDataError(ValueError):
    """The tables do not hold the expected set of reference games."""


def _reference_runs(tables_dir):
    """runId -> games.parquet row (dict) of the admissible Opus 4.6 Ground Control games."""
    g = pq.read_table(Path(tables_dir) / "games.parquet",
                      columns=["runId", "admissible", "agentModel", "scenarioId", "gitDescribe", "outcome"])
    runs = {}
    for row in g.to_pylist():
        # `is True` / `==` on a null gives no match, as the pandas filter `== True` does.
        if row["admissible"] is True and row["agentModel"] == MODEL and row["scenarioId"] == SCENARIO:
            runs[row["runId"]] = row
    # An explicit check, not an assert: it must hold under `python -O` too.
    if len(runs) != N_GAMES:
        raise ReferenceDataError(f"expected {N_GAMES} admissible {MODEL} {SCENARIO} games in "
                                 f"games.parquet, found {len(runs)}")
    return runs


def load_games(tables_dir):
    """runId -> (turn tags, is-gvp flags), one entry per tool_call row of each reference game.

    Rows are the `type == "tool_call"` rows of tool_calls.parquet, in stable `ts` order within
    each game (`ts` is monotonic in the native order of the 6 games, so the sort moves no row).
    The tag of a row is its `turn` column; a null turn counts as 0.

    Games without any tool_call row are left out. Games are returned in ascending runId order;
    the maximum R(H) keeps the first game on ties.
    """
    runs = _reference_runs(tables_dir)
    c = pq.read_table(Path(tables_dir) / "tool_calls.parquet", columns=["runId", "type", "ts", "turn", "tool"])
    rows = {rid: [] for rid in runs}
    cols = [c.column(k).to_pylist() for k in ("runId", "type", "ts", "turn", "tool")]
    for rid, typ, ts, turn, tool in zip(*cols):
        if typ == "tool_call" and rid in rows:
            rows[rid].append((ts, 0 if turn is None else int(turn), tool == GVP))
    games = {}
    for rid in sorted(rows):
        if not rows[rid]:
            continue  # a game without tool_call rows has no rate (as with a pandas groupby)
        s = sorted(rows[rid], key=lambda r: r[0])  # sorted() is stable
        games[rid] = ([r[1] for r in s], [r[2] for r in s])
    return games, runs


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    games, runs = load_games(argv[1])
    print(f"reference games: {len(games)}")
    rates = []
    for rid, (_, v) in games.items():
        g, n = sum(v), len(v)
        rates.append(Fraction(g, n))
        row = runs[rid]
        print(f"{rid}  {row['gitDescribe'] or '':26s} gvp {g:3d} / {n:5d} = {100 * g / n:.3f} %  {row['outcome']}")
    print(f"games with calls: {len(games)}")
    g_all = sum(sum(v) for _, v in games.values())
    n_all = sum(len(v) for _, v in games.values())
    print(f"pooled: {g_all} / {n_all} = {100 * g_all / n_all:.3f} %")
    print(f"MAX per-game rate: {100 * float(max(rates)):.3f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
