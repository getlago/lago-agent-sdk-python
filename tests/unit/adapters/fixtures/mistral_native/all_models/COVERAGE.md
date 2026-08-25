# Mistral native — captured-model coverage

Every model below was captured live and run through the adapter. 73 captures reduced to 12 committed fixtures: one per row of this table, chosen because the sweep tests assert DISPATCH and non-zero usage, so a second capture with the same key re-asserts the same thing.

Recapture with the `capture*.py` / `capture*.ts` script in this tree; the sweep tests skip cleanly when the directory is absent, so a missing capture reads as "not covered" rather than as a pass.

| Committed fixture | Distinguishing key | Models it stands for |
|---|---|---|
| `codestral-2508.json` | family `codestral` | `codestral-2508`, `codestral-latest` |
| `devstral-2512.json` | family `devstral` | `devstral-2512`, `devstral-latest`, `devstral-medium-2507`, `devstral-medium-latest`, `devstral-small-2507` |
| `magistral-medium-2509.json` | family `magistral` | `magistral-medium-2509`, `magistral-medium-latest`, `magistral-small-2509`, `magistral-small-latest` |
| `magistral-medium-2509__vision.json` | family `magistral` + vision call | `magistral-medium-2509`, `magistral-medium-latest`, `magistral-small-2509`, `magistral-small-latest` |
| `ministral-14b-2512.json` | family `ministral` | `ministral-14b-2512`, `ministral-14b-latest`, `ministral-3b-2512`, `ministral-3b-latest`, `ministral-8b-2512`, `ministral-8b-latest` |
| `ministral-14b-2512__vision.json` | family `ministral` + vision call | `ministral-14b-2512`, `ministral-14b-latest`, `ministral-3b-2512`, `ministral-3b-latest`, `ministral-8b-2512`, `ministral-8b-latest` |
| `mistral-large-2411.json` | family `mistral` | `mistral-large-2411`, `mistral-large-2512`, `mistral-large-latest`, `mistral-large-pixtral-2411`, `mistral-medium-2505`, `mistral-medium-2508`, `mistral-medium-2604`, `mistral-medium-3-5`, `mistral-medium-3.5`, `mistral-medium-3`, `mistral-medium-c21211-r0-75`, `mistral-medium-latest`, `mistral-medium`, `mistral-small-2506`, `mistral-small-2603`, `mistral-small-latest`, `mistral-tiny-2407`, `mistral-tiny-latest`, `mistral-vibe-cli-fast`, `mistral-vibe-cli-latest`, `mistral-vibe-cli-with-tools` |
| `mistral-large-2512__vision.json` | family `mistral` + vision call | `mistral-large-2512`, `mistral-large-latest`, `mistral-medium-2505`, `mistral-medium-2508`, `mistral-medium-2604`, `mistral-medium-3-5`, `mistral-medium-3.5`, `mistral-medium-3`, `mistral-medium-c21211-r0-75`, `mistral-medium-latest`, `mistral-medium`, `mistral-small-2506`, `mistral-small-2603`, `mistral-small-latest`, `mistral-vibe-cli-fast`, `mistral-vibe-cli-latest`, `mistral-vibe-cli-with-tools` |
| `open-mistral-nemo-2407.json` | family `open` | `open-mistral-nemo-2407`, `open-mistral-nemo` |
| `pixtral-large-2411.json` | family `pixtral` | `pixtral-large-2411` |
| `pixtral-large-latest__vision.json` | family `pixtral` + vision call | `pixtral-large-latest` |
| `voxtral-mini-2507.json` | family `voxtral` | `voxtral-mini-2507`, `voxtral-mini-latest`, `voxtral-small-2507`, `voxtral-small-latest` |
