# SPDX-License-Identifier: MIT
"""get_victory_progress rate of two Civilization VI arms against the CivBench reference, at an
equal turn horizon H.

    python measure.py CONTROL_DIR TREATMENT_DIR REFERENCE_TABLES_DIR [--prior-listing FILE]
    python measure.py --selftest

CONTROL_DIR and TREATMENT_DIR each hold one folder per take (`takeNN` is spelled `tomaNN`,
two digits), each with the Claude Code `stream.jsonl` of that take and, optionally, the
`sesion.jsonl` event log of the proxy. REFERENCE_TABLES_DIR holds `games.parquet` and
`tool_calls.parquet` of the dataset `civbench/civbench-v1` (read by reference.py). Read-only;
prints to stdout. Standard library only, plus pyarrow through reference.py. `--selftest` builds
synthetic cases in a temporary directory (removed afterwards), checks the rules below on them,
and exits 0 only if every case passes; it needs no data and no network.

Rules, as fixed before the control game was played:
  universe   every tool_use block of every stream.jsonl of every take (takes in name order),
             in take order and order of appearance, deduplicated by tool_use id across the arm.
  tag        turn_read(tool, text, is_error) of the LAST tool result received before the call
             for which it is not None; `text` is the text blocks joined by a newline, with the
             `{"result": "..."}` wrapper removed. A call before any turn was read is tagged 0.
             The tag follows the last turn read and may go down after a reload.
  T_fin      max(largest call tag, largest turn_read over ANY tool result of the arm).
  H          min(T_fin_treatment, T_fin_control) - 1.
  window     the calls BEFORE the first call tagged > H (all of them if none is). Same rule for
             control, treatment and each of the 6 reference games (reference tag: the `turn`
             column, null -> 0, rows in stable `ts` order).
  P, E, R    exact fractions g/N; R = maximum over the 6 reference games (a game with an empty
             window is skipped; ties keep the first game in runId order).
  row        a validity check fails, or H < T_MIN -> 4; else P <= R -> 1; R < P < E -> 2;
             P >= E -> 3.

Validity statuses: FAIL (fails on the data), OK (decided from the data), PARTIAL (the half
that the data can decide passes; the other half is not checked by this script) and UNDECIDED
(not decided by this script). Only FAIL moves the row; every PARTIAL and UNDECIDED item, and
the take-1 exception of 11.5 when it may apply, is printed after FINAL ROW as `not checked by
this script`.
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import re
import sys
import tempfile
from fractions import Fraction as F
from pathlib import Path

sys.dont_write_bytecode = True  # importing reference.py must not leave a __pycache__ behind

import reference  # noqa: E402

GVP = reference.GVP
BLOCK = "BLOCKED by Warden"  # the treatment's block marker; the control must never show it
STARTUP = 30  # a call with index < 30 inside its take is a start-up (orientation) call
T_MIN = 121  # fixed before the control was played
MODEL = reference.MODEL
CLAUDE_CODE_VERSION = "2.1.283"
N_CIV_TOOLS = 70
TREATMENT_ONLY_TOOLS = ("commit", "dispose", "ledger")
LAUNCHER = "civ_sinlua.bat"
SHA_SYSTEM_PROMPT = "1f4e9397c0bdfe2714e001f280022ce834f04c157f036425f941fb3f7fc7834e"
SHA_FIRST_MESSAGE = "b34fba3eda50bc04a41adb2fee505959a8cac06d7b57c95ca1b2418e49266ec3"
SENS_E_TAKE = "toma08"  # sensitivity E: the treatment without this take
TAKE_RE = re.compile(r"toma\d\d")

# Proxy log events that only a take with tool calls writes (V10: a take without stream.jsonl).
LOG_ACTIVITY = ("evaluacion", "resultado_end_turn", "consulta_estado")

END_TURN_TOOL = "end_turn"
OVERVIEW_TOOL = "get_game_overview"
# `Turn X -> Y` anywhere in an end_turn result: the turn reached is Y.
RE_TURN_XY = re.compile(r"Turn\s+(\d+)\s*->\s*(\d+)")
# First line of get_game_overview: `Turn N | ...` or `Turn N/M | ...` (M is the turn limit).
RE_TURN_OVERVIEW = re.compile(r"^Turn (\d+)(?:/\d+)? \|")

# What these scripts cannot decide from the data (protocol 11.10). A PARTIAL check leaves the
# half named here; the checks in NOT_CHECKED are not looked at at all.
UNCHECKED_HALF = {
    "V4": ("resumes", "that each resume message comes from genera_reanudacion.py over the control's own "
                      "diaries, written in the operator's control/ directory (not in the data)"),
    "V5": ("dd20190, run_lua", "that the harness is commit dd20190 and run_lua is inactive (not in the data; a "
                               "launcher with the same file name in another fork passes the half checked here)"),
    "V12": ("file dates", "that no loaded save file has a modification date earlier than the creation of the "
                          "control game (not in the data)"),
}
NOT_CHECKED = {
    "V6": "initial state: the two diffs of 11.3 empty",
    "V7": "operator interventions and save loads against the reload rule of 11.9 step 10",
    "V8": "isolated configuration: sha256 of settings.json and working directory (paths of the machine "
          "where the games ran)",
    "V15": "abandonment for a cause other than (a), (b) or (c) of 11.5",
}


def turn_read(tool, text, is_error=False):
    """The turn number the harness wrote in this tool result, or None.

    end_turn: the Y of the first `Turn X -> Y` in the text (the turn reached, not the one
    ended). get_game_overview: the N of a first line `Turn N | ...` or `Turn N/M | ...`.
    Any other tool, an error result, or a non-string text: None (the tag does not change).
    """
    if is_error or not isinstance(text, str):
        return None
    if tool == END_TURN_TOOL:
        m = RE_TURN_XY.search(text)
        return int(m.group(2)) if m else None
    if tool == OVERVIEW_TOOL:
        m = RE_TURN_OVERVIEW.match(text)
        return int(m.group(1)) if m else None
    return None


# ----------------------------------------------------------------------------- reading
def flat(content):
    """Tool result content as one string: text blocks joined by a newline."""
    if isinstance(content, list):
        return "\n".join(y.get("text", "") for y in content if isinstance(y, dict) and y.get("type") == "text")
    return content if isinstance(content, str) else str(content)


def unwrap(t):
    """Remove the `{"result": "..."}` wrapper that the Claude Code stream puts around MCP text."""
    if isinstance(t, str) and t.startswith("{"):
        try:
            o = json.loads(t)
        except ValueError:
            return t
        if isinstance(o, dict) and set(o) == {"result"} and isinstance(o["result"], str):
            return o["result"]
    return t


def json_lines(path):
    """(parsed objects, 1-based numbers of lines that do not parse, exempt partial last line?)

    The last line of a file that does not end in a newline, if it does not parse, is the write
    cut by the death of the process: it is dropped and reported apart, not counted as bad.
    """
    raw = Path(path).read_bytes()
    if not raw:
        return [], [], False  # an empty file has no line, cut or not
    parts = raw.split(b"\n")
    ends_nl = raw.endswith(b"\n")
    if ends_nl:
        parts = parts[:-1]
    objs, bad, exempt = [], [], False
    for k, p in enumerate(parts):
        try:
            objs.append(json.loads(p.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            if k == len(parts) - 1 and not ends_nl:
                exempt = True
            else:
                bad.append(k + 1)
    return objs, bad, exempt


def read_arm(base):
    base = Path(base)
    takes = sorted(d.name for d in base.iterdir() if TAKE_RE.fullmatch(d.name) and d.is_dir())
    calls, info = [], {}
    seen, names, inputs = set(), {}, {}
    dup = 0
    tag = 0
    res_max = 0
    n_res_turn = 0
    unwrapped = 0
    for tk in takes:
        d = base / tk
        it = {"init": None, "bad": [], "exempt": False, "no_stream": False, "n_calls": 0,
              "warden_res": [], "diary": [], "loads": [], "end_turn_res": 0, "warden_other": 0,
              "has_sesion": False, "log_bad": [], "log_exempt": False, "seals": 0, "log_activity": 0,
              "evals": 0, "evals_blocked": [], "advances": False}
        info[tk] = it
        # Proxy event log, read with the same tolerant reader as the stream (its hash chain is
        # not verified here). A line that parses but is not an object counts as a bad line.
        if (d / "sesion.jsonl").is_file():
            it["has_sesion"] = True
            events, it["log_bad"], it["log_exempt"] = json_lines(d / "sesion.jsonl")
            for k, e in enumerate(events):
                if not isinstance(e, dict):
                    it["log_bad"].append(f"object {k + 1} is not a JSON object")
                    continue
                tipo = e.get("tipo")
                if tipo in LOG_ACTIVITY:
                    it["log_activity"] += 1
                if tipo == "sello":
                    it["seals"] += 1
                if tipo == "evaluacion":
                    it["evals"] += 1
                    if e.get("blocked") is True:
                        it["evals_blocked"].append(e.get("turno"))
                if tipo == "resultado_end_turn" and e.get("avanzo"):
                    it["advances"] = True
        if not (d / "stream.jsonl").is_file():
            it["no_stream"] = True  # V10: whether this take had calls is not in the data
            continue
        objs, it["bad"], it["exempt"] = json_lines(d / "stream.jsonl")
        pending = False  # an end_turn was blocked and no get_victory_progress has followed yet
        idx = 0
        for o in objs:
            if not isinstance(o, dict):
                continue
            if o.get("type") == "system" and o.get("subtype") == "init" and it["init"] is None:
                it["init"] = o
            m = o.get("message")
            cont = m.get("content") if isinstance(m, dict) else None
            if not isinstance(cont, list):
                if "Warden" in json.dumps(o, ensure_ascii=False):
                    it["warden_other"] += 1
                continue
            for x in cont:
                if not isinstance(x, dict):
                    continue
                if o.get("type") == "assistant" and x.get("type") == "tool_use":
                    if x.get("id") in seen:
                        dup += 1
                        continue
                    seen.add(x["id"])
                    n = x["name"].split("__")[-1]
                    names[x["id"]] = n
                    inputs[x["id"]] = x.get("input")
                    forced = False
                    if n == GVP and pending:
                        forced, pending = True, False
                    calls.append({"take": tk, "idx": idx, "name": n, "tag": tag, "forced": forced})
                    idx += 1
                    it["n_calls"] += 1
                elif o.get("type") == "user" and x.get("type") == "tool_result":
                    i = x.get("tool_use_id")
                    n = names.get(i, "")
                    raw = flat(x.get("content"))
                    text = unwrap(raw)
                    unwrapped += text is not raw
                    err = bool(x.get("is_error"))
                    if n == END_TURN_TOOL:
                        pending = BLOCK in raw
                        it["end_turn_res"] += 1
                    if "Warden" in raw:
                        it["warden_res"].append((n, raw[:120]))
                    if n == "get_diary":
                        it["diary"].append(text[:300])
                    if n == "load_game_save":
                        # Whole text: the harness can prefix "Kill: ... | Launch: ... | Load: " before
                        # "Save loading (", and a cut here would hide a successful load from V12.
                        it["loads"].append((inputs.get(i), err, text))
                    t = turn_read(n, text, err)
                    if t is not None:
                        tag = t
                        n_res_turn += 1
                        res_max = max(res_max, t)
                elif o.get("type") == "user" and "Warden" in json.dumps(x, ensure_ascii=False):
                    it["warden_other"] += 1
    tf_calls = max((c["tag"] for c in calls), default=None)
    tfin = None if tf_calls is None else max(tf_calls, res_max)
    return {"base": str(base), "takes": takes, "calls": calls, "info": info, "dup": dup,
            "tfin": tfin, "tfin_calls": tf_calls, "tfin_res": res_max, "n_res_turn": n_res_turn,
            "unwrapped": unwrapped}


# ----------------------------------------------------------------------------- window and rates
def window_cut(tags, H):
    """Index of the first tag > H (len if none)."""
    for i, t in enumerate(tags):
        if t > H:
            return i
    return len(tags)


def window(items, tags, H):
    """The items inside the window of H: those before the first one tagged > H."""
    return items[:window_cut(tags, H)]


def arm_window(calls, H):
    return window(calls, [c["tag"] for c in calls], H)


def rate(calls, H):
    w = arm_window(calls, H)
    return sum(c["name"] == GVP for c in w), len(w)


def ref_rate(tags, gvp, H):
    w = window(gvp, tags, H)
    return sum(w), len(w)


def R(games, H):
    """(fraction, runId, g, N) of the reference game with the highest rate in the window of H."""
    best = None
    for rid, (tags, gvp) in games.items():
        g, n = ref_rate(tags, gvp, H)
        if n:
            r = F(g, n)
            if best is None or r > best[0]:
                best = (r, rid, g, n)
    return best


# ----------------------------------------------------------------------------- verdicts
def row_of(P, Rv, Ev):
    if P <= Rv:
        return 1
    if P < Ev:
        return 2
    return 3


def fr(g, n):
    return f"{g}/{n}" + (f" = {100 * g / n:.4f} %" if n else " (N=0)")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def file_name(p):
    """Last component of a Windows or POSIX path, so that no directory of the machine is printed."""
    return re.split(r"[\\/]", str(p))[-1]


def init_fails(init):
    """Which of V1, V2, V13 the `init` of one take with calls fails."""
    tools = init.get("tools") or []
    civ = [t for t in tools if t.startswith("mcp__civ__")]
    wt = [t for t in civ if t.split("__")[-1] in TREATMENT_ONLY_TOOLS]
    out = []
    if init.get("model") != MODEL:
        out.append("V1")
    if wt or len(civ) != N_CIV_TOOLS:
        out.append("V2")
    if init.get("claude_code_version") != CLAUDE_CODE_VERSION:
        out.append("V13")
    return out


def log_problem(it):
    """Why the proxy log of one take cannot be used, or None if it can.

    Not usable: no sesion.jsonl; a line that does not parse (other than the exempt cut last
    line); no `sello` event (the proxy writes it when the session closes); or end_turn results
    in the stream and no `evaluacion` in the log. Used by V3 (its evaluations cannot be looked
    at) and by sensitivity B (whether the take advances is not decided).
    """
    if not it["has_sesion"]:
        return "no sesion.jsonl"
    if it["log_bad"]:
        return f"lines that do not parse {it['log_bad'][:5]}"
    if not it["seals"]:
        return "no sello"
    if it["end_turn_res"] and not it["evals"]:
        return f"{it['end_turn_res']} end_turn results in the stream and 0 evaluacion"
    return None


def save_key(name):
    """What V12 compares: the file name of a save, without `.Civ6Save`, in lower case."""
    base = file_name(str(name).strip())
    if base.lower().endswith(".civ6save"):
        base = base[:-len(".civ6save")]
    return base.lower()


def validity(C, prior_listing):
    """The validity checks over the control data: (id, status, detail, todo).

    status is FAIL, OK, PARTIAL or UNDECIDED (module docstring). todo is None when nothing is
    left to check, '' when the whole check is left, or the name of the half that is left.
    V9 (H < T_MIN) is added by the caller.
    """
    out = []
    full = [tk for tk in C["takes"] if C["info"][tk]["n_calls"] >= 1]
    inits = {tk: C["info"][tk]["init"] or {} for tk in full}
    bad = [tk for tk in full if inits[tk].get("model") != MODEL]
    out.append(("V1", "FAIL" if bad else "OK", f"takes with full init {len(full)}; model other than {MODEL}: "
                f"{[(tk, inits[tk].get('model')) for tk in bad]}", None))
    det = []
    for tk in full:
        tools = inits[tk].get("tools") or []
        civ = [t for t in tools if t.startswith("mcp__civ__")]
        wt = [t for t in civ if t.split("__")[-1] in TREATMENT_ONLY_TOOLS]
        if wt or len(civ) != N_CIV_TOOLS:
            det.append((tk, len(civ), wt))
    out.append(("V2", "FAIL" if det else "OK", f"(take, n mcp__civ__, treatment-only tools) out of rule "
                f"({N_CIV_TOOLS} and none): {det}", None))
    wres = [(tk, x) for tk in C["takes"] for x in C["info"][tk]["warden_res"]]
    eb = [(tk, t) for tk in C["takes"] for t in C["info"][tk]["evals_blocked"]]
    nb = sum(1 for _, x in wres if BLOCK in x[1])
    # A take that has (or may have) calls and no usable proxy log: its evaluations cannot be looked at.
    no_log = [(tk, log_problem(C["info"][tk])) for tk in C["takes"]
              if (C["info"][tk]["n_calls"] >= 1 or C["info"][tk]["no_stream"]) and log_problem(C["info"][tk])]
    no_log_tk = [tk for tk, _ in no_log]
    log_exempt = [tk for tk in C["takes"] if C["info"][tk]["log_exempt"]]
    v3_fail = bool(wres or eb)
    out.append(("V3", "FAIL" if v3_fail else ("PARTIAL" if no_log else "OK"),
                f"results with 'Warden': {len(wres)} (with '{BLOCK}': {nb}); "
                f"evaluations blocked=true: {len(eb)} {eb[:12]}; takes without a usable sesion.jsonl (evaluations "
                f"not looked at): {no_log}; sesion.jsonl partial last line exempt (dropped) in: {log_exempt}; "
                f"(informative, not V3: other stream lines with 'Warden' outside a "
                f"tool_result: {sum(C['info'][tk]['warden_other'] for tk in C['takes'])})",
                f"evaluations of {no_log_tk}" if no_log else None))
    det, missing = [], []
    for tk in C["takes"]:
        p = Path(C["base"]) / tk / "entrada_system_prompt.md"
        if p.is_file():
            if sha(p) != SHA_SYSTEM_PROMPT:
                det.append((tk, "prompt", sha(p)[:12]))
        elif tk in full:
            missing.append((tk, "entrada_system_prompt.md"))
    first = C["takes"][0] if C["takes"] else None
    msgs = sorted(Path(C["base"], first).glob("entrada_mensaje_*.md")) if first else []
    if len(msgs) != 1:
        missing.append((first, f"entrada_mensaje_*.md: {len(msgs)} files"))
    elif sha(msgs[0]) != SHA_FIRST_MESSAGE:
        det.append((first, msgs[0].name, sha(msgs[0])[:12]))
    out.append(("V4", "FAIL" if (det or missing) else "PARTIAL", f"hash differs: {det}; file missing: {missing}",
                UNCHECKED_HALF["V4"][0]))
    det = []
    for tk in full:
        p = Path(C["base"]) / tk / "mcp.json"
        try:
            args = json.loads(p.read_text(encoding="utf-8"))["mcpServers"]["civ"]["args"]
        except (OSError, ValueError, KeyError, TypeError):
            det.append((tk, "mcp.json unreadable"))
            continue
        name = file_name(args[-1]) if args else ""
        if name.lower() != LAUNCHER:
            det.append((tk, name))
    out.append(("V5", "FAIL" if det else "PARTIAL", f"harness launcher file in mcp.json is not {LAUNCHER}: {det}",
                UNCHECKED_HALF["V5"][0]))
    for vid in ("V6", "V7", "V8"):
        out.append((vid, "UNDECIDED", "", ""))
    # V10 speaks of takes with calls. Bad lines fail it in a take with calls; in a take without
    # any call that parses, whether a bad line was a call is not in the data. A take without
    # stream.jsonl fails it when its proxy log records activity that only calls produce.
    info = C["info"]
    det = [(tk, "lines that do not parse", info[tk]["bad"][:5]) for tk in C["takes"]
           if info[tk]["bad"] and info[tk]["n_calls"] >= 1]
    ghost = [(tk, info[tk]["log_activity"]) for tk in C["takes"] if info[tk]["no_stream"] and info[tk]["log_activity"]]
    bad_no_calls = [tk for tk in C["takes"] if info[tk]["bad"] and info[tk]["n_calls"] == 0]
    exempt = [tk for tk in C["takes"] if info[tk]["exempt"]]
    no_stream = [tk for tk in C["takes"] if info[tk]["no_stream"] and not info[tk]["log_activity"]]
    todo = "; ".join(([f"takes without stream.jsonl {no_stream}"] if no_stream else [])
                     + ([f"takes without calls with lines that do not parse {bad_no_calls}"] if bad_no_calls else []))
    out.append(("V10", "FAIL" if (det or ghost) else ("UNDECIDED" if todo else "OK"),
                f"{det}; partial last line exempt (dropped) in: {exempt}; takes without stream.jsonl whose "
                f"sesion.jsonl records {'/'.join(LOG_ACTIVITY)} (take, events): {ghost}; takes without stream.jsonl "
                f"and no such event (whether these had calls is not in the data): {no_stream}; takes without calls "
                f"with lines that do not parse (whether those lines were calls is not in the data): {bad_no_calls}",
                todo or None))
    dia = [(tk, x[:100]) for tk in C["takes"] for x in C["info"][tk]["diary"]
           if not x.startswith("No diary entries yet")]
    ndia = sum(len(C["info"][tk]["diary"]) for tk in C["takes"])
    # Entries are allowed if written by the control itself; their dates are not in the stream.
    out.append(("V11", "UNDECIDED" if dia else "OK", f"get_diary results {ndia}; with entries (dates to check "
                f"against the creation of the control): {len(dia)} {dia[:5]}", "diary entry dates" if dia else None))
    loads = [(tk, inp, txt) for tk in C["takes"] for inp, err, txt in C["info"][tk]["loads"]
             if not err and "Save loading (" in txt and "FAILED" not in txt]
    prior = Path(prior_listing).read_text(encoding="utf-8", errors="replace") if prior_listing else None
    # Compared by file name, without `.Civ6Save` and without case: a listed `auto\prev\AutoSave_0001.Civ6Save`
    # matches a load of `autosave_0001`, and a load of `AutoSave_000` matches nothing.
    prior_keys = {save_key(x) for x in prior.splitlines() if x.strip()} if prior is not None else set()
    det = []
    for tk, inp, _ in loads:
        name = json.dumps(inp, ensure_ascii=False)
        if "ctrl_t1" in name.lower():
            continue  # the control's own initial save and its copy (11.6, V12)
        cand = [str(v) for v in (inp or {}).values()] if isinstance(inp, dict) else []
        if prior is not None and any(v and save_key(v) in prior_keys for v in cand):
            det.append((tk, name, "IN THE PRIOR LISTING"))
    out.append(("V12", "FAIL" if det else ("PARTIAL" if prior is not None else "UNDECIDED"),
                f"successful load_game_save: {len(loads)} "
                f"{[(tk, json.dumps(i, ensure_ascii=False)) for tk, i, _ in loads][:8]}; in the prior listing "
                f"(file name, case-insensitive): {det}; "
                f"prior listing {'read' if prior is not None else 'NOT given (--prior-listing)'}",
                UNCHECKED_HALF["V12"][0] if prior is not None else ""))
    bad = [(tk, inits[tk].get("claude_code_version")) for tk in full
           if inits[tk].get("claude_code_version") != CLAUDE_CODE_VERSION]
    out.append(("V13", "FAIL" if bad else "OK", f"version other than {CLAUDE_CODE_VERSION}: {bad}", None))
    out.append(("V14", "FAIL" if not C["calls"] else "OK", f"control calls: {len(C['calls'])}", None))
    out.append(("V15", "UNDECIDED", "", ""))
    return out


TAKE1_ITEM = "take-1 exception (11.5)"


def first_take_notice(C):
    """Protocol 11.6, precedence (1): the take-1 exception of 11.5, as (notice, may apply?), or None.

    The exception removes take 1 from the control if its init fails V1, V2 or V13 and the take
    was killed before its stream had the result of its first end_turn. With 1 or more end_turn
    results in its stream it does not apply. With 0 it may: when the take was killed is not in
    the data, so this script does not apply it, counts the take, and leaves it to check (tail of
    FINAL ROW).
    """
    if not C["takes"]:
        return None
    t1 = C["takes"][0]
    it = C["info"][t1]
    if it["n_calls"] < 1:
        return None
    v = init_fails(it["init"] or {})
    if not v:
        return None
    head = (f"NOTICE take-1 exception (11.5; 11.6 precedence (1)): {t1} fails {v}; end_turn results in its "
            f"stream: {it['end_turn_res']}.")
    if it["end_turn_res"]:
        return (f"{head} The exception does not apply (it needs 0): {t1} is part of the control and its "
                f"failing checks stand", False)
    return (f"{head} The exception may apply (it also needs the take to have been killed before its first "
            f"end_turn result, which is not in the data); not checked by this script, which counts {t1} as part "
            f"of the control", True)


def not_checked(vs):
    """`V4 (resumes), V5 (...), V6, ...`: everything the validity block leaves to check."""
    items = []
    for vid, _, _, todo in vs:
        if todo is not None:
            items.append(vid if todo == "" else f"{vid} ({todo})")
    return ", ".join(items)


def print_validity(vs):
    for vid, st, det, todo in vs:
        if vid in NOT_CHECKED:
            print(f"  {vid} {st}: not checked by this script: {NOT_CHECKED[vid]}")
            continue
        extra = ""
        if todo and vid in UNCHECKED_HALF and todo == UNCHECKED_HALF[vid][0]:
            extra = f"; NOT CHECKED ({todo}): {UNCHECKED_HALF[vid][1]}"
        elif todo == "":
            extra = "; not decided by this script"
        print(f"  {vid} {st}: {det}{extra}")


# ----------------------------------------------------------------------------- report
def decompose(calls, H):
    """Window of H, its queries, and sensitivity D: forced / start-up (index < 30) / rest."""
    w = arm_window(calls, H)
    gv = [c for c in w if c["name"] == GVP]
    forced = [c for c in gv if c["forced"]]
    startup = [c for c in gv if not c["forced"] and c["idx"] < STARTUP]
    rest = [c for c in gv if not c["forced"] and c["idx"] >= STARTUP]
    return w, gv, forced, startup, rest


def cut_point(name, calls, H):
    tags = [c["tag"] for c in calls]
    k = window_cut(tags, H)
    if k == len(tags):
        print(f"CUT {name}: no call tagged > {H}; all {len(tags)} count")
        return
    c = calls[k]
    late = [x for x in calls[k:] if x["tag"] <= H]
    print(f"CUT {name}: first call tagged > {H} = #{k + 1} ({c['take']}, index {c['idx']}, "
          f"{c['name']}, tag {c['tag']}); later calls tagged <= {H}: {len(late)} "
          f"(queries {sum(x['name'] == GVP for x in late)}); outside the window: {len(calls) - k}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("control_dir", nargs="?")
    ap.add_argument("treatment_dir", nargs="?")
    ap.add_argument("reference_dir", nargs="?", help="directory with games.parquet and tool_calls.parquet")
    ap.add_argument("--prior-listing", default=None,
                    help="text file with the save listing taken before the control started (V12)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic cases in a temporary directory and exit (no data needed)")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.control_dir and a.treatment_dir and a.reference_dir):
        ap.error("CONTROL_DIR, TREATMENT_DIR and REFERENCE_TABLES_DIR are required")
    Cc = read_arm(a.control_dir)
    E2 = read_arm(a.treatment_dir)
    games, _ = reference.load_games(a.reference_dir)
    for nom, X in (("CONTROL", Cc), ("TREATMENT", E2)):
        print(f"{nom}: takes {len(X['takes'])}, tool_use {len(X['calls'])}, gvp "
              f"{sum(c['name'] == GVP for c in X['calls'])}, forced {sum(c['forced'] for c in X['calls'])}, "
              f"duplicate ids dropped {X['dup']}, unwrapped results {X['unwrapped']}")
        print(f"{nom} T_fin = {X['tfin']} (largest call tag {X['tfin_calls']}; largest turn_read of "
              f"any result {X['tfin_res']}, over {X['n_res_turn']} results with a turn)")
    TE = E2["tfin"]
    table = {}
    print("TABLE H | E(H) | R(H) game | E>R")
    for H in range(100, TE + 1):
        g, n = rate(E2["calls"], H)
        r = R(games, H)
        table[H] = (F(g, n), r[0])
        print(f"TABLE {H} | {g}/{n} | {r[2]}/{r[3]} {r[1][:30]} | {F(g, n) > r[0]}")
    tmin = None
    for H in range(TE - 1, 119, -1):
        if table[H][0] > table[H][1]:
            tmin = H
        else:
            break
    print(f"T_min computed = {tmin} (fixed: {T_MIN}; {'matches' if tmin == T_MIN else 'DOES NOT MATCH'}); "
          f"H in [120,{TE - 1}] with E <= R: {[H for H in range(120, TE) if not table[H][0] > table[H][1]]}")
    vs = validity(Cc, a.prior_listing)
    H = None if Cc["tfin"] is None else min(TE, Cc["tfin"]) - 1
    v9 = H is None or H < T_MIN
    vs.append(("V9", "FAIL" if v9 else "OK", f"H = {H}; T_min = {T_MIN}", None))
    vs.sort(key=lambda v: int(v[0][1:]))
    print("VALIDITY (FAIL / OK / PARTIAL: one half checked / UNDECIDED: not decided by this script):")
    print_validity(vs)
    notice = first_take_notice(Cc)
    if notice:
        print(notice[0])
    failing = [v[0] for v in vs if v[1] == "FAIL"]
    partial = [v[0] for v in vs if v[1] == "PARTIAL"]
    undecided = [v[0] for v in vs if v[1] == "UNDECIDED"]
    left = not_checked(vs)
    if notice and notice[1]:
        left = f"{left}, {TAKE1_ITEM}" if left else TAKE1_ITEM
    tail = f"; not checked by this script: {left}" if left else ""
    print(f"FAILING: {failing}; PARTIAL: {partial}; UNDECIDED: {undecided}")
    if H is None:
        print(f"FINAL ROW: 4 (V14: the control has no calls; no T_fin_C and no P(H)){tail}")
        return 0
    print(f"H = min(T_fin_treatment = {TE}, T_fin_control = {Cc['tfin']}) - 1 = {H}")
    gP, nP = rate(Cc["calls"], H)
    gE, nE = rate(E2["calls"], H)
    r = R(games, H)
    if nP == 0 or nE == 0:
        print(f"P or E with N = 0 at H = {H} (P {gP}/{nP}, E {gE}/{nE}): no rate")
        print(f"FINAL ROW: 4 (N = 0){tail}")
        return 0
    P, Ev, Rv = F(gP, nP), F(gE, nE), r[0]
    print(f"P({H}) = {fr(gP, nP)}")
    print(f"E({H}) = {fr(gE, nE)}")
    print(f"R({H}) = {fr(r[2], r[3])} ({r[1]})")
    print(f"ARITH P vs R: {gP}*{r[3]}={gP * r[3]} vs {r[2]}*{nP}={r[2] * nP}; P vs E: {gP}*{nE}={gP * nE} vs "
          f"{gE}*{nP}={gE * nP}; E vs R: {gE}*{r[3]}={gE * r[3]} vs {r[2]}*{nE}={r[2] * nE}")
    part = Rv < Ev
    print(f"partition R(H) < E(H): {part}{'' if part else '  ** THE TABLE IS NOT A PARTITION AT THIS H **'}")
    fn = row_of(P, Rv, Ev)
    print(f"NUMERIC ROW (rows 1-3, validity not looked at): {fn}"
          f"{' (tie P = R)' if P == Rv else ''}{' (tie P = E)' if P == Ev else ''}")
    final = 4 if (failing or v9) else fn
    print(f"FINAL ROW: {final}" + (f" (because of {failing})" if final == 4 else "") + tail)
    gR = math.floor(Rv * nP) + 1
    gEf = math.ceil(Ev * nP)
    print(f"BOUNDARIES with N = N_C({H}) = {nP}: g = {gP}; g_R = floor(R*N)+1 = {gR} (leave row 1: "
          f"{'already out' if gP >= gR else f'{gR - gP} short'}); g_E = ceil(E*N) = {gEf} (reach row 3: "
          f"{'already in' if gP >= gEf else f'{gEf - gP} short'})")
    cut_point("CONTROL", Cc["calls"], H)
    cut_point("TREATMENT", E2["calls"], H)
    for rid, (tags, gvp) in games.items():
        k = window_cut(tags, H)
        if k < len(tags):
            print(f"CUT ref {rid[:30]}: first row with turn > {H} at position {k} (turn {tags[k]}); later "
                  f"with turn <= {H}: {sum(t <= H for t in tags[k:])}; window {len(window(gvp, tags, H))}")
        else:
            print(f"CUT ref {rid[:30]}: no row with turn > {H}; window {len(window(gvp, tags, H))}")
    print("SENSITIVITIES (same H, same tag, same window; they do not change the row):")
    sens = {}
    for nom, X in (("CONTROL", Cc), ("TREATMENT", E2)):
        w, gv, forced, startup, rest = decompose(X["calls"], H)
        startup_all = [c for c in gv if c["idx"] < STARTUP]
        # B: whether a take with end_turn results advances is read from its proxy log; without a
        # usable log (log_problem) it is not decided, and the take is named, not silently dropped.
        b_undecided = sorted(tk for tk in X["takes"]
                             if X["info"][tk]["end_turn_res"] and log_problem(X["info"][tk]))
        adv = sorted(tk for tk in X["takes"] if X["info"][tk]["advances"] and tk not in b_undecided)
        wb = [c for c in w if c["take"] in adv]
        tk_in = sorted(set(c["take"] for c in w))
        resumed = [tk for tk in X["takes"] if any(Path(X["base"], tk).glob("entrada_mensaje_reanuda_*.md"))]
        sens[nom] = {"A": (len(gv) - len(startup_all), len(w)), "B": (sum(c["name"] == GVP for c in wb), len(wb)),
                     "B_undecided": b_undecided}
        print(f"  A {nom}: without queries of index < {STARTUP}: {fr(*sens[nom]['A'])}")
        print(f"  B {nom}: only takes that advance {adv}: {fr(*sens[nom]['B'])}; takes with end_turn results "
              f"and no usable sesion.jsonl (not decided, left out): {b_undecided}")
        print(f"  D {nom}: forced / start-up / rest = {len(forced)} / {len(startup)} / {len(rest)} "
              f"(= {len(forced) + len(startup) + len(rest)} of {len(gv)}); takes with calls in the window "
              f"{len(tk_in)}; takes {len(X['takes'])}; takes with a resume message {len(resumed)}")
        if nom == "TREATMENT":
            we = [c for c in w if c["take"] != SENS_E_TAKE]
            sens[nom]["E"] = (sum(c["name"] == GVP for c in we), len(we))
            print(f"  E TREATMENT: without {SENS_E_TAKE}: {fr(*sens[nom]['E'])} ({SENS_E_TAKE} in the window: "
                  f"{sum(c['take'] == SENS_E_TAKE for c in w)})")
    for s in ("A", "B"):
        (gp, npp), (ge, ne) = sens["CONTROL"][s], sens["TREATMENT"][s]
        und = [f"{arm.lower()} {tk}" for arm in ("CONTROL", "TREATMENT") for tk in sens[arm].get(f"{s}_undecided", [])]
        note = f"  NOT DECIDED: holds only with {und} left out" if und else ""
        if npp and ne:
            f_ = row_of(F(gp, npp), Rv, F(ge, ne))
            print(f"  row with sensitivity {s}: {f_}{'  FRAGILE' if f_ != fn else ''}{note}")
        elif und:
            print(f"  row with sensitivity {s}: none (N = 0){note}")
    ge, ne = sens["TREATMENT"]["E"]
    f_ = row_of(P, Rv, F(ge, ne))
    print(f"  row with sensitivity E: {f_}{'  FRAGILE' if f_ != fn else ''}")
    print("DESCRIPTIVE (does not decide), whole game, uncut:")
    for nom, X in (("CONTROL", Cc), ("TREATMENT", E2)):
        print(f"  {nom}: {fr(sum(c['name'] == GVP for c in X['calls']), len(X['calls']))}")
    full = max((F(sum(v), len(v)), sum(v), len(v), rid) for rid, (_, v) in games.items())
    print(f"  reference (per-game maximum, uncut): {fr(full[1], full[2])} ({full[3]})")
    for rid, (_, v) in games.items():
        print(f"    {rid}: {sum(v)}/{len(v)}")
    return 0


# ----------------------------------------------------------------------------- self-test
# Synthetic data only: nothing here reads the experiment's data or the network.
_NL = chr(10)
_PROMPT = "synthetic system prompt" + _NL
_FIRST = "synthetic first message" + _NL


def _use(i, name, inp=None):
    return {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": i, "name": "mcp__civ__" + name,
                                                          "input": inp or {}}]}}


def _res(i, text, is_error=False):
    # Wrapped as the Claude Code stream wraps MCP text.
    return {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": i, "is_error": is_error,
                                                     "content": [{"type": "text",
                                                                  "text": json.dumps({"result": text})}]}]}}


def _init(model=MODEL, version=CLAUDE_CODE_VERSION, tools=None):
    return {"type": "system", "subtype": "init", "model": model, "claude_code_version": version,
            "tools": [f"mcp__civ__tool{k}" for k in range(N_CIV_TOOLS)] if tools is None else tools}


def _take(root, name, steps, *, init=None, launcher="C:/harness/" + LAUNCHER, sesion=(), sesion_raw=None,
          stream=True, first_message=False, tail=b"", bad_line=False, prompt=_PROMPT, first=_FIRST):
    """One synthetic take.

    steps: (tool, text), (id, tool, text), or a dict with `tool`, `text` and optionally `id`,
    `input`, `error`; a None text means no result. bad_line puts a line that does not parse
    right after the init. sesion_raw, when given, is written as sesion.jsonl byte for byte.
    """
    d = Path(root) / name
    d.mkdir(parents=True)
    rows = [init or _init()]
    for k, s in enumerate(steps):
        if isinstance(s, dict):
            i, tool, text, inp, err = s.get("id", f"{name}-{k}"), s["tool"], s.get("text"), s.get("input"), \
                s.get("error", False)
        else:
            i, tool, text = s if len(s) == 3 else (f"{name}-{k}", s[0], s[1])
            inp, err = None, False
        rows.append(_use(i, tool, inp))
        if text is not None:
            rows.append(_res(i, text, err))
    if stream:
        lines = [json.dumps(x) + _NL for x in rows]
        if bad_line:
            lines.insert(1, "{bad" + _NL)
        (d / "stream.jsonl").write_bytes("".join(lines).encode("utf-8") + tail)
    # Bytes, not text mode: the hashes of V4 must not depend on the platform's newline.
    if sesion_raw is not None:
        (d / "sesion.jsonl").write_bytes(sesion_raw)
    elif sesion is not None:
        (d / "sesion.jsonl").write_bytes(_jsonl(sesion))
    (d / "entrada_system_prompt.md").write_bytes(prompt.encode("utf-8"))
    if first_message:
        (d / "entrada_mensaje_ground_control.md").write_bytes(first.encode("utf-8"))
    mcp = {"mcpServers": {"civ": {"command": "python", "args": ["proxy.py", "--", launcher]}}}
    (d / "mcp.json").write_bytes(json.dumps(mcp).encode("utf-8"))
    return d


def _jsonl(rows):
    return "".join(json.dumps(e) + _NL for e in rows).encode("utf-8")


def _simple(g, n, tf):
    """n calls: an overview (tag 0), g queries, n - 2 - g other calls, and an end_turn reaching tf."""
    return ([("get_game_overview", "Turn 1 | Babylon")] + [(GVP, "progress")] * g
            + [("get_units", "units")] * (n - 2 - g) + [("end_turn", f"Turn 1 -> {tf}")])


def _plus(extra, g=1, n=10, tf=122):
    """_simple(g, n, tf) with the steps `extra` inserted before its final end_turn."""
    s = _simple(g, n, tf)
    return s[:-1] + list(extra) + s[-1:]


_SEAL = {"tipo": "sello"}
_LOG = [{"tipo": "evaluacion", "blocked": False, "turno": 1}, {"tipo": "resultado_end_turn", "avanzo": True}, _SEAL]


def _arm(root, takes):
    """takes: list of (steps, options) for toma01, toma02, ...; toma01 carries the first message."""
    Path(root).mkdir(parents=True)
    for k, (steps, opts) in enumerate(takes):
        o = {"sesion": _LOG, **opts}
        _take(root, f"toma{k + 1:02d}", steps, first_message=(k == 0), **o)
    return root


def _reference(root, n_games=6):
    """Six admissible games ref-a..ref-f of 10 tool_call rows each; ref-b and ref-e tie at 1/10.

    ref-f has its rows out of `ts` order and one null turn: sorted, its tags are 0, 3, 5, 1 x 7.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(root)
    root.mkdir(parents=True)
    ids = ["ref-a", "ref-b", "ref-c", "ref-d", "ref-e", "ref-f"][:n_games]
    games = {"runId": ids + ["ref-z"], "admissible": [True] * len(ids) + [False],
             "agentModel": [MODEL] * (len(ids) + 1), "scenarioId": [reference.SCENARIO] * (len(ids) + 1),
             "gitDescribe": ["synthetic"] * (len(ids) + 1), "outcome": ["none"] * (len(ids) + 1)}
    pq.write_table(pa.table(games), root / "games.parquet")
    cols = {"runId": [], "type": [], "ts": [], "turn": [], "tool": []}

    def row(rid, typ, ts, turn, tool):
        for k, v in zip(cols, (rid, typ, ts, turn, tool)):
            cols[k].append(v)
    for rid in ids + ["ref-z"]:
        if rid == "ref-f":
            for ts, turn in ((30, 5), (10, None), (20, 3)):
                row(rid, "tool_call", ts, turn, "get_units")
            for ts in range(40, 110, 10):
                row(rid, "tool_call", ts, 1, "get_units")
            continue
        for k in range(10):
            gvp = rid in ("ref-b", "ref-e", "ref-z") and k == 5
            row(rid, "tool_call", k, 1, GVP if gvp else "get_units")
        row(rid, "result", 99, 1, GVP)  # not a tool_call: never counted
    pq.write_table(pa.table({"runId": pa.array(cols["runId"], pa.string()), "type": pa.array(cols["type"]),
                             "ts": pa.array(cols["ts"], pa.int64()), "turn": pa.array(cols["turn"], pa.int64()),
                             "tool": pa.array(cols["tool"])}), root / "tool_calls.parquet")
    return root


def _run(argv):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = main(argv)
    except Exception as e:  # a crash is a failed case, not an aborted self-test
        return None, buf.getvalue() + f"{_NL}EXCEPTION {type(e).__name__}: {e}"
    return rc, buf.getvalue()


def selftest():
    """Synthetic cases for the rules of the module docstring; exit status 0 only if all pass."""
    global SHA_SYSTEM_PROMPT, SHA_FIRST_MESSAGE
    fails, n = [], [0]

    def check(cid, cond, detail=""):
        n[0] += 1
        print(f"{'PASS' if cond else 'FAIL'} {cid}: {detail}")
        if not cond:
            fails.append(cid)

    saved = (SHA_SYSTEM_PROMPT, SHA_FIRST_MESSAGE)
    SHA_SYSTEM_PROMPT = hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest()
    SHA_FIRST_MESSAGE = hashlib.sha256(_FIRST.encode("utf-8")).hexdigest()
    try:
        with tempfile.TemporaryDirectory(prefix="measure_selftest_") as tmp:
            ctx = {"tmp": Path(tmp), "check": check}
            for group in _SELFTEST_GROUPS:
                # An exception inside a group is one failed case; the other groups still run.
                try:
                    group(ctx)
                except Exception as e:
                    check(f"group_{group.__name__.lstrip('_')}", False, f"EXCEPTION {type(e).__name__}: {e}")
    finally:
        SHA_SYSTEM_PROMPT, SHA_FIRST_MESSAGE = saved
    print(f"SELFTEST {'OK' if not fails else 'FAILED'}: {n[0] - len(fails)}/{n[0]} checks pass; failing: {fails}")
    return 1 if fails else 0


# Statuses of a control with every file in rule and a prior listing, and the tail of its FINAL ROW.
_BASE_STATUS = {"V1": "OK", "V2": "OK", "V3": "OK", "V4": "PARTIAL", "V5": "PARTIAL", "V6": "UNDECIDED",
                "V7": "UNDECIDED", "V8": "UNDECIDED", "V9": "OK", "V10": "OK", "V11": "OK", "V12": "PARTIAL",
                "V13": "OK", "V14": "OK", "V15": "UNDECIDED"}
_BASE_TAIL = ["V4 (resumes)", "V5 (dd20190, run_lua)", "V6", "V7", "V8", "V12 (file dates)", "V15"]


def _tail(*extra, drop=()):
    """_BASE_TAIL with the items `extra` (in V order; the take-1 item last) and without `drop`."""
    items = [x for x in _BASE_TAIL if x not in drop] + [x for x in extra if x.startswith("V")]
    items.sort(key=lambda x: int(x.split()[0][1:]))
    return items + [x for x in extra if not x.startswith("V")]


def _expect(check, cid, out, row, over=None, tail=None, detail=""):
    """The exact status of every `  Vn STATUS:` line, V1-V15, and the whole FINAL ROW line."""
    want = dict(_BASE_STATUS, **(over or {}))
    got = dict(re.findall(r"^  (V\d+) (\w+):", out, re.M))
    diff = {v: (got.get(v), want.get(v)) for v in sorted(set(got) | set(want), key=lambda s: int(s[1:]))
            if got.get(v) != want.get(v)}
    line = f"FINAL ROW: {row}; not checked by this script: {', '.join(_BASE_TAIL if tail is None else tail)}"
    rows = re.findall(r"^FINAL ROW: .*$", out, re.M)
    check(cid, not diff and rows == [line],
          f"{detail}; statuses differing (got, want): {diff}; FINAL ROW {rows} want [{line!r}]"
          + ("" if "EXCEPTION" not in out else " " + out.strip().splitlines()[-1]))


def _st_setup(ctx):
    tmp = ctx["tmp"]
    ctx["ref"] = _reference(tmp / "ref")
    prior = tmp / "prior.txt"
    prior.write_bytes(_NL.join(["auto\\prev\\AutoSave_0001.Civ6Save", "CTRL_T1.Civ6Save", ""]).encode("utf-8"))
    ctx["prior"] = prior
    ctx["treat"] = _arm(tmp / "t_2of10", [(_simple(2, 10, 122), {})])

    def scenario(name, takes, treatment=None, listing=True):
        c = _arm(tmp / name, takes)
        argv = [str(c), str(treatment or ctx["treat"]), str(ctx["ref"])]
        return _run(argv + (["--prior-listing", str(prior)] if listing else []))
    ctx["scenario"] = scenario


def _st_units(ctx):
    tmp, check = ctx["tmp"], ctx["check"]
    # --- turn_read, unit table (tool, text, is_error, expected)
    unit = [("end_turn", "Turn 5 -> 6" + _NL + "ok", False, 6),
            ("end_turn", "Turn 5->6", False, 6),
            ("end_turn", "Turn 5 -> 6", True, None),
            ("end_turn", "no header here", False, None),
            ("end_turn", None, False, None),
            ("get_game_overview", "Turn 7 | Babylon", False, 7),
            ("get_game_overview", "Turn 7/330 | Babylon", False, 7),
            ("get_game_overview", "Era: x" + _NL + "Turn 7 | Babylon", False, None),
            ("get_victory_progress", "Turn 5 -> 6", False, None)]
    bad = [(t, x, e, turn_read(t, x, e), want) for t, x, e, want in unit if turn_read(t, x, e) != want]
    check("turn_read_table", not bad, f"{len(unit)} cases; failing (tool, text, is_error, got, want): {bad}")
    got = flat([{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}])
    check("flat_newline", got == "a" + _NL + "b", repr(got))

    # --- tag, dedup across takes, T_fin (read_arm)
    arm = _arm(tmp / "tags", [
        ([("get_game_overview", "Turn 4 | Babylon"), ("get_units", "units"), (GVP, "progress"),
          ("u-dup", "end_turn", "Turn 4 -> 5")], {}),
        ([("u-dup", "end_turn", None), ("get_units", "units"), ("end_turn", "Turn 5 -> 6")], {})])
    a = read_arm(arm)
    tags = [c["tag"] for c in a["calls"]]
    check("tag_follows_last_turn_read", tags == [0, 4, 4, 4, 5, 5],
          f"tags {tags}, want [0, 4, 4, 4, 5, 5] (a result without a turn keeps the tag; take 2 inherits it)")
    check("dedup_across_takes", (a["dup"], len(a["calls"])) == (1, 6),
          f"duplicates dropped {a['dup']}, calls {len(a['calls'])}; want 1 and 6 (u-dup of toma01 re-emitted in toma02)")
    check("tfin_last_result", a["tfin"] == 6, f"T_fin {a['tfin']}, want 6 (last result Turn 5 -> 6, no call after)")

    # --- window: before the first tag > H, nothing after it
    calls = [{"name": GVP if k in (1, 5) else "x", "tag": t} for k, t in enumerate([0, 1, 5, 5, 6, 2])]
    check("window_first_above_H", rate(calls, 5) == (1, 4), f"rate at H=5 {rate(calls, 5)}, want (1, 4)")

    # --- rows and their ties
    check("row_ties", (row_of(F(1, 10), F(1, 10), F(2, 10)), row_of(F(2, 10), F(1, 10), F(2, 10))) == (1, 3),
          "P = R -> 1, P = E -> 3")

    # --- lines that do not parse: only a cut LAST line without newline is exempt; an empty file has none
    p = tmp / "lines.jsonl"
    p.write_bytes(b'{"a": 1}' + b"\n" + b"{bad" + b"\n" + b'{"b": 2}' + b"\n" + b"{cut")
    r1 = json_lines(p)[1:]
    p.write_bytes(b'{"a": 1}' + b"\n" + b"{bad" + b"\n")
    r2 = json_lines(p)[1:]
    p.write_bytes(b"")
    r3 = json_lines(p)
    check("v10_exempt_last_line_only", (r1, r2, r3) == (([2], True), ([2], False), ([], [], False)),
          f"cut last line {r1}, want ([2], True); bad last line WITH newline {r2}, want ([2], False); "
          f"empty file {r3}, want ([], [], False)")


def _st_reference(ctx):
    tmp, check = ctx["tmp"], ctx["check"]
    games, _ = reference.load_games(ctx["ref"])
    check("reference_order_and_null", games.get("ref-f", ([], []))[0] == [0, 3, 5] + [1] * 7,
          f"ref-f tags {games.get('ref-f', ([], []))[0]}, want [0, 3, 5, 1 x 7] (sorted by ts, null -> 0)")
    check("reference_tool_call_rows", [len(v[1]) for v in games.values()] == [10] * 6,
          f"rows per game {[len(v[1]) for v in games.values()]}")
    try:
        reference.load_games(_reference(tmp / "ref5", n_games=5))
        raised = "no error"
    except reference.ReferenceDataError as e:
        raised = f"ReferenceDataError: {e}"
    check("reference_game_count", raised.startswith("ReferenceDataError"), raised)


def _st_rows(ctx):
    """End to end: rows, ties, boundaries, the V9 frontier, T_min."""
    check, scenario = ctx["check"], ctx["scenario"]
    tail = "; not checked by this script: " + ", ".join(_BASE_TAIL)

    rc, out = scenario("c_tie_R", [(_simple(1, 10, 122), {})])
    check("tie_P_eq_R_row1", "NUMERIC ROW (rows 1-3, validity not looked at): 1 (tie P = R)" in out
          and "FINAL ROW: 1;" in out, "P = 1/10 = R -> row 1")
    check("R_tie_first_game", "R(121) = 1/10 = 10.0000 % (ref-b)" in out, "ref-b and ref-e tie; ref-b first")
    check("boundaries_integer_product", "g_R = floor(R*N)+1 = 2 " in out and "g_E = ceil(E*N) = 2 " in out,
          "R*N = 1 and E*N = 2 are integers: g_R = 2, g_E = 2")
    check("v9_frontier_121_passes", "H = min(T_fin_treatment = 122, T_fin_control = 122) - 1 = 121" in out
          and "  V9 OK: H = 121;" in out, "T_fin 122 in both arms -> H = 121, V9 OK")
    check("final_row_tail", "FINAL ROW: 1" + tail + _NL in out, f"FINAL ROW ends with '{tail[2:]}'")
    check("partial_statuses", all(f"  {v} PARTIAL:" in out for v in ("V4", "V5", "V12"))
          and "  V3 OK:" in out, "V4, V5, V12 PARTIAL with a prior listing; V3 OK with every sesion.jsonl")
    check("undecided_line_not_empty", "UNDECIDED: ['V6', 'V7', 'V8', 'V15']" in out, "V6, V7, V8, V15 always listed")
    _expect(check, "all_statuses_in_rule", out, "1", detail="every file in rule: the base statuses")
    check("tmin_strict_on_data", "T_min computed = 120 (fixed: 121; DOES NOT MATCH); H in [120,121] with E <= R: []"
          in out, "E = 2/10 > R = 1/10 at H = 121 and 120")

    rc, out = scenario("c_tie_E", [(_simple(2, 10, 122), {})])
    check("tie_P_eq_E_row3", "NUMERIC ROW (rows 1-3, validity not looked at): 3 (tie P = E)" in out
          and "FINAL ROW: 3;" in out, "P = 2/10 = E -> row 3")

    rc, out = scenario("c_h120", [(_simple(1, 10, 121), {})])
    check("v9_frontier_120_row4", "- 1 = 120" in out and "FINAL ROW: 4 (because of ['V9'])" in out,
          "T_fin_C 121 -> H = 120 -> row 4 by V9 only")
    _expect(check, "v9_fail_statuses", out, "4 (because of ['V9'])", {"V9": "FAIL"})

    # M-2: E = R at H = 121 -> T_min is not reached (strict E > R); >= would give 120
    t_eq = _arm(ctx["tmp"] / "t_1of10", [(_simple(1, 10, 122), {})])
    rc, out = scenario("c_tmin_eq", [(_simple(1, 10, 122), {})], treatment=t_eq)
    check("tmin_E_eq_R_breaks", "T_min computed = None (fixed: 121; DOES NOT MATCH); H in [120,121] with E <= R: "
          "[120, 121]" in out, "E = 1/10 = R at H = 121: no T_min")


def _st_validity(ctx):
    """One case per FAIL branch (I-1), each with the status of every V line and the whole tail."""
    check, scenario = ctx["check"], ctx["scenario"]
    civ = [f"mcp__civ__tool{k}" for k in range(N_CIV_TOOLS)]

    rc, out = scenario("c_v1", [(_simple(1, 10, 122), {"init": _init(model="claude-other")})])
    check("failing_v_row4", "NUMERIC ROW (rows 1-3, validity not looked at): 1" in out
          and "FINAL ROW: 4 (because of ['V1'])" in out, "numeric row 1, V1 fails -> row 4")
    check("take1_exception_notice", "NOTICE take-1 exception" in out and "end_turn results in its stream: 1." in out,
          "toma01 fails V1: the notice names it with its end_turn results")
    check("take1_notice_does_not_apply", "The exception does not apply (it needs 0)" in out
          and "not checked" not in out.split("NOTICE", 1)[-1].split(_NL, 1)[0],
          "1 end_turn result: the notice says the exception does not apply")
    _expect(check, "v1_fail_statuses", out, "4 (because of ['V1'])", {"V1": "FAIL"})

    rc, out = scenario("c_v2_69", [(_simple(1, 10, 122), {"init": _init(tools=civ[:-1])})])
    _expect(check, "v2_fail_69_tools", out, "4 (because of ['V2'])", {"V2": "FAIL"}, detail="69 mcp__civ__ tools")
    rc, out = scenario("c_v2_commit", [(_simple(1, 10, 122), {"init": _init(tools=civ[:-1] + ["mcp__civ__commit"])})])
    _expect(check, "v2_fail_treatment_tool", out, "4 (because of ['V2'])", {"V2": "FAIL"},
            detail="70 tools, one of them commit")

    rc, out = scenario("c_v3_block", [(_plus([("end_turn", BLOCK + ": look first")]), {})])
    _expect(check, "v3_fail_blocked_result", out, "4 (because of ['V3'])", {"V3": "FAIL"},
            detail=f"an end_turn result with '{BLOCK}'")

    rc, out = scenario("c_v4_prompt", [(_simple(1, 10, 122), {"prompt": "another system prompt" + _NL})])
    _expect(check, "v4_fail_prompt_hash", out, "4 (because of ['V4'])", {"V4": "FAIL"}, detail="other system prompt")
    rc, out = scenario("c_v4_first", [(_simple(1, 10, 122), {"first": "another first message" + _NL})])
    _expect(check, "v4_fail_first_message_hash", out, "4 (because of ['V4'])", {"V4": "FAIL"},
            detail="other first message")

    rc, out = scenario("c_v10_bad", [(_simple(1, 10, 122), {"bad_line": True})])
    _expect(check, "v10_fail_bad_line_with_calls", out, "4 (because of ['V10'])", {"V10": "FAIL"},
            detail="a line that does not parse in a take with calls")
    # M-5: bad lines in a take without calls do not fail V10; whether they were calls is not in the data
    rc, out = scenario("c_v10_bad_nocalls", [(_simple(1, 10, 122), {}), ([], {"bad_line": True})])
    _expect(check, "v10_bad_line_without_calls_undecided", out, "1", {"V10": "UNDECIDED"},
            _tail("V10 (takes without calls with lines that do not parse ['toma02'])"))
    # I-4: no stream.jsonl, and a proxy log with an evaluacion: the take had calls -> V10 fails
    rc, out = scenario("c_v10_ghost", [(_simple(1, 10, 122), {}), ([], {"stream": False})])
    _expect(check, "v10_fail_nostream_with_log_activity", out, "4 (because of ['V10'])", {"V10": "FAIL"},
            detail="toma02 without stream.jsonl, its sesion.jsonl has an evaluacion")
    rc, out = scenario("c_v10_nostream_seal", [(_simple(1, 10, 122), {}),
                                               ([], {"stream": False, "sesion": [{"tipo": "herencia"}, _SEAL]})])
    _expect(check, "v10_nostream_log_without_activity_undecided", out, "1", {"V10": "UNDECIDED"},
            _tail("V10 (takes without stream.jsonl ['toma02'])"))

    # V11: an entry in a get_diary result is left to check (its date is not in the stream)
    rc, out = scenario("c_v11", [(_plus([("get_diary", "No diary entries yet."),
                                          ("get_diary", "[T5] 2026-09-20: campus planned")]), {})])
    _expect(check, "v11_entries_undecided", out, "1", {"V11": "UNDECIDED"}, _tail("V11 (diary entry dates)"))

    # V12: by file name and case; failed loads do not count; CTRL_T1 is exempt
    def load(name, text="Save loading (5s). Steps: ok", error=False):
        return {"tool": "load_game_save", "text": text, "input": {"save_name": name}, "error": error}
    rc, out = scenario("c_v12_listed", [(_plus([load("D:/saves/AUTOSAVE_0001.Civ6Save")]), {})])
    _expect(check, "v12_fail_listed_by_file_name", out, "4 (because of ['V12'])", {"V12": "FAIL"},
            detail="auto\\prev\\AutoSave_0001.Civ6Save listed; AUTOSAVE_0001.Civ6Save loaded from another directory")
    rc, out = scenario("c_v12_noext", [(_plus([load("autosave_0001")]), {})])
    _expect(check, "v12_fail_listed_without_extension", out, "4 (because of ['V12'])", {"V12": "FAIL"},
            detail="AutoSave_0001.Civ6Save listed; autosave_0001 loaded, as the harness names saves")
    rc, out = scenario("c_v12_prefix", [(_plus([load("AutoSave_000"), load("0_MCP_0001")]), {})])
    _expect(check, "v12_prefix_not_listed", out, "1", detail="AutoSave_000 is a prefix of a listed name, not a name")
    check("v12_counts_successful_loads", "successful load_game_save: 2 " in out, "two successful loads")
    rc, out = scenario("c_v12_failed", [(_plus([load("AutoSave_0001", "Save loading (5s) FAILED: no file"),
                                                load("AutoSave_0001", error=True),
                                                load("AutoSave_0001", "Kill: game killed")]), {})])
    _expect(check, "v12_failed_loads_not_counted", out, "1",
            detail="a listed save whose load FAILED, errored or never started")
    check("v12_failed_loads_count_zero", "successful load_game_save: 0 " in out, "no successful load")
    long_prefix = "Kill: " + "x" * 250 + " | Launch: ok | Load: Save loading (5s). Steps: ok"
    rc, out = scenario("c_v12_long", [(_plus([load("AutoSave_0001", long_prefix)]), {})])
    _expect(check, "v12_fail_listed_after_long_prefix", out, "4 (because of ['V12'])", {"V12": "FAIL"},
            detail="'Save loading (' after a 250+ character Kill/Launch prefix still counts as a load")
    rc, out = scenario("c_v12_ctrl", [(_plus([load("ctrl_t1")]), {})])
    _expect(check, "v12_ctrl_t1_exempt", out, "1", detail="CTRL_T1 is in the listing and exempt")
    check("v12_ctrl_t1_counted", "successful load_game_save: 1 " in out, "the load is counted, then exempted")

    rc, out = scenario("c_v13", [(_simple(1, 10, 122), {"init": _init(version="2.1.282")})])
    _expect(check, "v13_fail_version", out, "4 (because of ['V13'])", {"V13": "FAIL"}, detail="Claude Code 2.1.282")

    rc, out = scenario("c_v14", [([], {})])
    _expect(check, "v14_fail_no_calls", out, "4 (V14: the control has no calls; no T_fin_C and no P(H))",
            {"V9": "FAIL", "V14": "FAIL"}, detail="one take, init only")

    rc, out = scenario("c_no_listing", [(_simple(1, 10, 122), {})], listing=False)
    check("v12_undecided_without_listing", "  V12 UNDECIDED:" in out and ", V12, V15" + _NL in out,
          "no --prior-listing: V12 UNDECIDED and listed whole")
    _expect(check, "v12_no_listing_statuses", out, "1", {"V12": "UNDECIDED"}, _tail("V12", drop=("V12 (file dates)",)))

    # M-3: init completo = at least one tool_use; a take without calls is not looked at
    rc, out = scenario("c_init_nocalls", [(_simple(1, 10, 122), {}),
                                          ([], {"init": _init(model="claude-other", version="0", tools=[])})])
    _expect(check, "init_complete_needs_a_call", out, "1",
            detail="toma02 has an init out of rule and no tool_use: V1, V2, V13 do not look at it")
    check("init_complete_count", "takes with full init 1;" in out, "1 of 2 takes has a full init")


def _st_notice(ctx):
    """I-2: the take-1 exception may apply only with 0 end_turn results."""
    check, scenario = ctx["check"], ctx["scenario"]
    rc, out = scenario("c_take1_zero", [([("get_game_overview", "Turn 1 | Babylon"), ("get_units", "units")],
                                         {"init": _init(model="claude-other")}),
                                        (_simple(1, 10, 122), {})])
    check("take1_notice_may_apply", "end_turn results in its stream: 0. The exception may apply" in out,
          "toma01 fails V1 with 0 end_turn results")
    _expect(check, "take1_exception_in_tail", out, "4 (because of ['V1'])", {"V1": "FAIL"}, _tail(TAKE1_ITEM),
            detail="the exception is left to check at the end of FINAL ROW")


def _st_log(ctx):
    """I-3 and I-5: a take with end_turn results and no usable proxy log."""
    check, scenario = ctx["check"], ctx["scenario"]
    v3_tail = _tail("V3 (evaluations of ['toma01'])")
    cases = [("empty", {"sesion_raw": b""}, "no sello"),
             ("no_seal", {"sesion": _LOG[:2]}, "no sello"),
             ("no_eval", {"sesion": [{"tipo": "herencia"}, _SEAL]}, "1 end_turn results in the stream and 0 evaluacion"),
             ("cut", {"sesion_raw": _jsonl(_LOG[:2]) + b'{"tipo": "sel'}, "no sello"),
             ("bad", {"sesion_raw": _jsonl(_LOG[:1]) + b"{bad" + _NL.encode() + _jsonl(_LOG[1:])},
              "lines that do not parse [2]"),
             ("nondict", {"sesion_raw": _jsonl([[1, 2]] + _LOG)},
              "lines that do not parse ['object 1 is not a JSON object']")]
    for name, opts, why in cases:
        rc, out = scenario(f"c_log_{name}", [(_simple(1, 10, 122), opts)])
        _expect(check, f"log_{name}_v3_partial", out, "1", {"V3": "PARTIAL"}, v3_tail,
                detail=f"sesion.jsonl {name}: V3 PARTIAL, not an abort")
        check(f"log_{name}_reason_and_B", f"('toma01', {why!r})" in out
              and "(not decided, left out): ['toma01']" in out
              and "  row with sensitivity B: none (N = 0)  NOT DECIDED: holds only with ['control toma01'] left out"
              in out, f"reason '{why}'; B names toma01 as not decided")
    rc, out = scenario("c_log_cut_exempt", [(_simple(1, 10, 122), {"sesion_raw": _jsonl(_LOG) + b'{"tipo": "x'})])
    _expect(check, "log_cut_last_line_exempt", out, "1", detail="a cut last line after the sello is exempt")
    check("log_cut_last_line_reported", "sesion.jsonl partial last line exempt (dropped) in: ['toma01']" in out,
          "the exempt line is reported")


def _st_sensitivities(ctx):
    tmp, check, scenario = ctx["tmp"], ctx["check"], ctx["scenario"]
    # --- A and D: forced first query after a block; index 30 is not start-up
    steps = ([("get_game_overview", "Turn 1 | Babylon"), ("end_turn", BLOCK + ": look first"), (GVP, "p"),
              ("end_turn", BLOCK + ": look first"), ("end_turn", "busy, try again"), (GVP, "p")]
             + [("get_units", "units")] * 24 + [(GVP, "p"), ("end_turn", "Turn 1 -> 122")])
    t_forced = _arm(tmp / "t_forced", [(steps, {})])
    rc, out = scenario("c_for_forced", [(_simple(1, 10, 122), {})], treatment=t_forced)
    check("sens_A_all_index_below_30", "  A TREATMENT: without queries of index < 30: 1/32 " in out,
          "queries at index 2 (forced), 5 and 30: A keeps 1 of 32")
    check("sens_D_forced_startup_rest", "  D TREATMENT: forced / start-up / rest = 1 / 1 / 1 " in out,
          "a block, then a non-blocked end_turn: the next query is not forced")

    # --- B: only takes whose log has a resultado_end_turn with avanzo true
    t1 = [("get_game_overview", "Turn 1 | Babylon"), (GVP, "p")] + [("get_units", "u")] * 7 \
        + [("end_turn", "Turn 1 -> 60")]
    t2 = [("get_game_overview", "Turn 60 | Babylon")] + [("get_units", "u")] * 4 + [("end_turn", "Turn 60 -> 122")]
    no_adv = [{"tipo": "evaluacion", "blocked": False, "turno": 60}, {"tipo": "resultado_end_turn", "avanzo": False},
              _SEAL]
    rc, out = scenario("c_sens_B", [(t1, {}), (t2, {"sesion": no_adv})])
    check("sens_B_only_advancing", "  B CONTROL: only takes that advance ['toma01']: 1/10 = 10.0000 %; takes with "
          "end_turn results and no usable sesion.jsonl (not decided, left out): []" in out,
          "toma02 has avanzo false: B keeps toma01, 1 of 10")

    # --- E: the treatment without toma08
    t8 = [("get_game_overview", "Turn 50 | Babylon"), (GVP, "p")] + [("get_units", "u")] * 3 \
        + [("end_turn", "Turn 50 -> 122")]
    tE = _arm(tmp / "t_E", [(_simple(2, 10, 50), {})] + [([], {})] * 6 + [(t8, {})])
    rc, out = scenario("c_sens_E", [(_simple(1, 10, 122), {})], treatment=tE)
    check("sens_E_without_toma08", "  E TREATMENT: without toma08: 2/10 = 20.0000 % (toma08 in the window: 6)" in out,
          "E = 3/16 with toma08, 2/10 without")

    # --- missing stream, missing proxy log, launcher in another fork, launcher of another name
    rc, out = scenario("c_fork_nostream", [(_simple(1, 10, 122), {"launcher": "D:/other-fork/" + LAUNCHER,
                                                                  "sesion": None}),
                                           ([], {"stream": False, "sesion": None})])
    check("nostream_no_abort", rc == 0 and "FINAL ROW:" in out, f"rc {rc}")
    check("nostream_v10_undecided", "  V10 UNDECIDED:" in out and "V10 (takes without stream.jsonl ['toma02'])" in out,
          "toma02 has no stream.jsonl")
    check("fork_v5_partial", "  V5 PARTIAL:" in out and "  V5 OK" not in out, "same file name in another fork")
    check("v3_evaluations_undecided_without_log", "  V3 PARTIAL:" in out and "V3 (evaluations of ['toma01', 'toma02'])" in out,
          "toma01 has calls and no sesion.jsonl; toma02 may have had calls")
    rc, out = scenario("c_v5_fail", [(_simple(1, 10, 122), {"launcher": "D:/private-dir/other.bat"})])
    check("v5_fail_file_name_only", "  V5 FAIL:" in out and "other.bat" in out and "private-dir" not in out,
          "the launcher's directory is not printed")

    # --- V3: a blocked evaluation in the proxy log fails it
    c = _arm(tmp / "c_evalblk", [(_simple(1, 10, 122), {"sesion": [{"tipo": "evaluacion", "blocked": True,
                                                                     "turno": 1}, _SEAL]})])
    v3 = [v for v in validity(read_arm(c), None) if v[0] == "V3"][0]
    check("v3_blocked_evaluation_fails", v3[1] == "FAIL", f"V3 {v3[1]}")


_SELFTEST_GROUPS = (_st_setup, _st_units, _st_reference, _st_rows, _st_validity, _st_notice, _st_log,
                    _st_sensitivities)


if __name__ == "__main__":
    sys.exit(main())
