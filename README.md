# Seetrex Warden on CivBench: pilot materials

Measurement scripts, rulesets and figure of the technical report *Seetrex Warden: an
external obligation gate for a long-horizon LLM agent. A pre-registered pilot (n = 1 per arm)
on the Ground Control scenario of CivBench (arXiv:2609.02459)* (pre-registered in the sense of
the report: commit dates in a private repository; no third-party timestamp) — https://doi.org/10.5281/zenodo.23528742.

The pilot played two games of Civilization VI with Claude Opus 4.6 on the Ground Control
scenario of CivBench (arXiv:2609.02459): one with Seetrex Warden, an MCP proxy that rejects
`end_turn` while an obligation is overdue, and one control game without it. One game per
arm: the report describes two games and supports no inference.

- `publicacion/medida/` — the scripts that compute the report's rates and verdict row from
  the data package and the CivBench dataset (MIT); not the counterfactual of §5.3 nor the
  ledger verification, which need the unpublished engine. Their README says how to run them.
- `rulesets/` — the two ruleset files: `warden_v2.json` (comparison game) and
  `control_vacio.json` (control game, empty). Rules say what to look at and what must be
  closed, never what move to make. Both files are published byte for byte: the
  `ruleset_hash` recorded in every evaluation of the ledgers is the SHA-256 of the whole file
  text encoded as a JSON string literal (in Python, `json.dumps(text, ensure_ascii=False)`
  encoded as UTF-8), not the SHA-256 of the file's bytes; for `warden_v2.json` it is
  `27a9c576…401acf`, the value printed in the report's appendix H. The `description` field of
  `warden_v2.json` names files, tasks and games of the authors' private repository; they are
  not published.
- `figure/` — Figure 1 of the report, SVG and PNG, with its caption. Figure 1 is drawn by a
  script that reads the complete internal logs; that script is not published.
- Release `v1.0` carries the report PDF; the canonical copy is the Zenodo record above.

The data package (agent logs of both games; CC BY 4.0; the game text it contains is not
covered, see its README) is on Hugging Face: https://huggingface.co/datasets/seetrex/seetrex-warden-civbench-pilot. The post: https://tabares.eu/seetrex-warden-civbench-pilot.
Scenario, harness and reference games come from CivBench, its repository `lmwilki/civ6-mcp`
and its dataset `civbench/civbench-v1`.

The Seetrex Warden engine and the ledger verifier are not published. Verification of the
ledgers is possible under agreement. Seetrex Warden is unrelated to the `warden-agent` MCP
proxy and to other projects named "Warden".
