# Measurement scripts: get_victory_progress rate, control vs treatment vs CivBench

These scripts compute the numbers of the control experiment in the technical report: how often
an LLM agent (Claude Opus 4.6) playing Civilization VI calls `get_victory_progress`, over all
its tool calls, up to the same turn horizon `H` in three arms:

- **treatment**: one game played with Seetrex Warden, a layer that makes the agent check its
  victory progress;
- **control**: one game with the same setup and no Seetrex Warden, except for the differences
  declared in the report (§4.2);
- **reference**: the 6 admissible Opus 4.6 games of the Ground Control scenario in the CivBench
  dataset (`civbench/civbench-v1`, arXiv:2609.02459); `R(H)` is the highest per-game rate.

They are self-contained: Python standard library plus `pyarrow` to read the parquet tables.
They import nothing from the Seetrex Warden product.

## Files

| File | What it is |
|---|---|
| `measure.py` | The measurement. Reads both arms and the reference, prints every figure and the verdict row. `--selftest` runs its synthetic cases. |
| `reference.py` | Reads the CivBench tables. Imported by `measure.py`; run alone it prints the uncut rate of each reference game. |
| `LICENSE` | MIT. |

## How to run

```
pip install pyarrow
python measure.py CONTROL_DIR TREATMENT_DIR REFERENCE_TABLES_DIR [--prior-listing FILE]
python reference.py REFERENCE_TABLES_DIR
python measure.py --selftest
```

Tested with Python 3.12 and pyarrow 20.0.0. Output goes to stdout. A measurement run writes
nothing (`measure.py` turns off the bytecode cache before importing `reference.py`, so no
`__pycache__` is left either). All paths are arguments.

**Self-test.** `python measure.py --selftest` needs no data and no network: it builds synthetic
arms and synthetic reference tables in a temporary directory, removed at the end, and checks
the rules below on them: the ties `P = R` (row 1) and `P = E` (row 3), the `H >= 121` frontier
(`H = 121` passes, `H = 120` is row 4), `T_min` with `E = R` inside its range, `g_R` and `g_E`
when `R*N` and `E*N` are integers, the tag and `T_fin` rules, the window, deduplication across
takes, the V10 exemption (only a cut last line without a newline), the reference order, null
turns and game count, sensitivities A, B, D and E, the definition of a full init (at least one
`tool_use`), the take-1 notice with 0 and with 1 `end_turn` results, proxy logs that cannot be
used (empty, cut, with a bad line, without `sello`, without `evaluacion`), and the FAIL branch
of every check the script decides (V1, V2, V3, V4, V10, V12, V13, V14), plus the UNDECIDED and
PARTIAL cases of the table below. In most end-to-end cases it checks the exact status of all 15
`Vn` lines and the whole `FINAL ROW` line; the rest check the specific line they are about. It prints one PASS/FAIL line per case and exits 0
only if all pass; an exception inside a group of cases counts as one failed case and the other
groups still run.

## Data expected

**`CONTROL_DIR` and `TREATMENT_DIR`**: one sub-folder per take (a take is one run of the agent
until it stopped or was stopped), named `tomaNN` with two digits, read in name order. In each:

| File | Used for | Required |
|---|---|---|
| `stream.jsonl` | Claude Code `--output-format stream-json` log: every tool call and tool result. The whole metric comes from here. | yes (a take without it contributes nothing; V10 fails if its `sesion.jsonl` records `evaluacion`, `resultado_end_turn` or `consulta_estado`, and is undecided otherwise) |
| `sesion.jsonl` | The proxy's event log, read as plain JSON lines with the same reader as the stream (a line that does not parse is reported, not fatal; a cut last line without a newline is exempt): `resultado_end_turn` with `avanzo` (sensitivity B), `evaluacion` with `blocked` (V3), `sello`. | no, but see *usable log* below |
| `entrada_system_prompt.md`, `entrada_mensaje_*.md` | The system prompt and the first message given to the agent (V4 hashes; resume messages counted in D). | for V4 |
| `mcp.json` | The MCP launch config (V5). | for V5 |

A take's log is **usable** if the file exists, every line parses (the exempt cut last line
aside), it has a `sello` event, and, if the take's stream has `end_turn` results, it has at
least one `evaluacion`. A take with calls (or without `stream.jsonl`) and no usable log leaves
the evaluation half of V3 unchecked (PARTIAL); a take with `end_turn` results and no usable
log is named as *not decided* in sensitivity B instead of being left out silently.

**`REFERENCE_TABLES_DIR`**: a directory with `games.parquet` and `tool_calls.parquet` from the
`civbench/civbench-v1` dataset. The reference games are those with `admissible == True`,
`agentModel == "claude-opus-4-6"` and `scenarioId == "ground_control"`; if there are not
exactly 6, `reference.py` stops with an error.

**`--prior-listing`**: a text file with the save-game listing taken before the control started,
one save per line; a successful `load_game_save` naming a save in it fails V12. Names are
compared by file name, without the directory, the `.Civ6Save` extension or case (a listed
`auto\prev\AutoSave_0001.Civ6Save` matches a load of `autosave_0001`; a load of `AutoSave_000`
matches nothing). Without it, V12 is undecided.

## What it measures

- **Universe:** every `tool_use` block of every take's `stream.jsonl`, in take order and order
  of appearance, deduplicated by `tool_use` id across the arm. Reloads, errors, calls with no
  result, takes that do not advance and replayed turns all count.
- **Turn tag of a call:** `turn_read(tool, text, is_error)` of the last tool result received
  before the call for which it is not `None`:
  - from `end_turn`, the **Y** of the first `Turn X -> Y` in the text (the turn reached);
  - from `get_game_overview`, the N of a first line `Turn N | ...` or `Turn N/M | ...`;
  - any other tool, or a result with `is_error`, leaves the tag unchanged.
  The text is the result's text blocks joined by a newline, with the `{"result": "..."}`
  wrapper of the stream removed. A call before any turn was read is tagged 0. The tag follows
  the last turn read and can go down after a reload.
- **Reference tag:** the `turn` column of `tool_calls.parquet` (null counts as 0), rows in
  stable `ts` order.
- **End of an arm:** `T_fin` = max of the largest call tag and the largest `turn_read` over any
  tool result of the arm.
- **Horizon:** `H = min(T_fin_treatment, T_fin_control) - 1`.
- **Window**, the same rule for control, treatment and each reference game: the calls **before
  the first call tagged `> H`** (all of them if none is). Nothing after that point counts, even
  if its tag goes back to `<= H`.
- **Rates**, as exact fractions: `P(H)` control, `E(H)` treatment, `R(H)` the highest of the 6
  reference games (a game with an empty window is skipped).
- **Verdict row:** a validity check fails, or `H < T_min = 121` -> row 4; otherwise
  `P <= R` -> row 1; `R < P < E` -> row 2; `P >= E` -> row 3. It also prints `g`, and the queries
  the control would need to leave row 1 (`g_R = floor(R*N) + 1`) and to reach row 3
  (`g_E = ceil(E*N)`).
- **Table `E(H)` and `R(H)`** for `H` = 100 to `T_fin_treatment`, and `T_min` recomputed from it
  (the smallest `H >= 120` such that `E(H') > R(H')` for every `H'` up to `T_fin_treatment - 1`);
  the fixed value 121 decides.
- **Sensitivities** at the same `H`: A (without queries whose index inside their take is < 30),
  B (only takes that advance, from a usable `sesion.jsonl`; takes without one are listed as not
  decided and the row of B says so), D (forced / start-up / rest decomposition,
  takes with calls in the window, takes, takes with a resume message) and E (treatment without
  take `toma08`). A, B and E each print the row they would give; D is a decomposition and has
  no row.
- **Descriptive, does not decide:** the uncut rate of each arm and of each reference game.

Results obtained with these scripts on the experiment's data: `H = 255`, `P(255) = 4/3728`,
`E(255) = 21/4236`, `R(255) = 5/2059` (steep-vermil-trebuchet-27), row 1, `g = 4`, `g_R = 10`,
`g_E = 19`; the row line ends with `not checked by this script: V4 (resumes), V5 (dd20190,
run_lua), V6, V7, V8, V12 (file dates), V15`.

## Validity checks

Every check V1-V15 of the protocol gets one line, with one of four statuses:

- **FAIL**: it fails on the data. Any FAIL makes the row 4.
- **OK**: decided from the data, and it passes.
- **PARTIAL**: the half that the data can decide passes; the line names the half that is not
  checked.
- **UNDECIDED**: not decided by this script.

Only FAIL moves the row. The `FINAL ROW` line ends with `not checked by this script:` and every
PARTIAL half and UNDECIDED check (and the take-1 exception when it may apply, below), so a row
printed here holds only while those do not fail.

| Check | What the script decides | Left out |
|---|---|---|
| V1 | model in the `init` of every take with a full init (at least one `tool_use`) | |
| V2 | 70 `mcp__civ__` tools and none of `commit`, `dispose`, `ledger` (same takes) | |
| V3 | no tool result with `Warden`, no `evaluacion` with `blocked` true | the evaluations of a take with calls or without `stream.jsonl` and no usable log (then PARTIAL) |
| V4 | sha256 of the system prompt and of the first message (the hashes of this experiment) | that the resume messages come from the generator (always PARTIAL) |
| V5 | the launcher file in `mcp.json` is named `civ_sinlua.bat` (on a FAIL, only the file name is printed) | the harness commit `dd20190` and `run_lua` inactive (always PARTIAL): a launcher of the same name in another fork passes |
| V6, V7, V8, V15 | nothing (initial state, operator interventions, isolated configuration, abandonment) | all (UNDECIDED) |
| V9 | `H >= 121` | |
| V10 | stream lines that do not parse in a take with calls (a cut last line without a newline is exempt); a take without `stream.jsonl` whose `sesion.jsonl` records `evaluacion`, `resultado_end_turn` or `consulta_estado` | whether a take without `stream.jsonl` and without such events had calls, and whether bad lines in a take without calls were calls (then UNDECIDED) |
| V11 | no diary entries visible to the control | the dates of the entries, if any (then UNDECIDED) |
| V12 | no successfully loaded save is in `--prior-listing` (by file name; `CTRL_T1` exempt) | the modification dates of the loaded files (PARTIAL); everything without `--prior-listing` (UNDECIDED) |
| V13 | Claude Code version (takes with a full init) | |
| V14 | the control has calls | |

The script also prints a notice when take 1 fails V1, V2 or V13 (the take-1 exception of the
protocol). With 1 or more `end_turn` results in its stream the exception does not apply, and
the notice says so. With 0 it may apply, depending on when the take was killed, which is not in
the data: the script counts the take, says so, and adds `take-1 exception (11.5)` to the end of
the `FINAL ROW` line.

## What it does NOT measure

- **Secondary C** (the counterfactual: what the Seetrex Warden rules would have found in the
  control's records, which had no Seetrex Warden) and **the verification of the proxy ledger** (its seal and hash chain) need the
  Seetrex Warden engine, which is part of the product and is not published. For the same reason the
  count of **missing turns with turn `<= H`** of secondary C, which the protocol asks to publish,
  is not here: it goes with C. `sesion.jsonl` is read as plain JSON lines; its integrity is not
  checked.
- **Checks that the data cannot answer:** the UNDECIDED checks and the PARTIAL halves of the
  table above. The script prints them; it does not decide them. The report states which were
  checked by other means and which remain pending. V8 depends on paths of the machine where
  the games ran.
- **No inference:** n = 1 game per arm. The scripts print fractions and rows, not p-values or
  intervals.

## Known limits

Found in review and not fixed, because none changes anything with the published data:

- A take whose stream has only lines that do not parse and no `sesion.jsonl` leaves V3 at OK;
  V10 names that take as UNDECIDED and the `FINAL ROW` tail lists it.
- A take with `end_turn` results whose usable log has no `resultado_end_turn` counts as "does
  not advance" in sensitivity B, and when B's window is empty its row line is not printed.
