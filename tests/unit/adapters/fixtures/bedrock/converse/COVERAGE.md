# Bedrock Converse — captured-model coverage

Every model below was captured live and run through the adapter. 39 captures reduced to 12 committed fixtures: one per row of this table, chosen because the sweep tests assert DISPATCH and non-zero usage, so a second capture with the same key re-asserts the same thing.

Recapture with the `capture*.py` / `capture*.ts` script in this tree; the sweep tests skip cleanly when the directory is absent, so a missing capture reads as "not covered" rather than as a pass.

| Committed fixture | Distinguishing key | Models it stands for |
|---|---|---|
| `eu.amazon.nova-lite-v1_0.json` | usage shape + pricing provider `amazon` | `eu.amazon.nova-2-lite-v1:0`, `eu.amazon.nova-lite-v1:0`, `eu.amazon.nova-micro-v1:0`, `eu.amazon.nova-pro-v1:0` |
| `eu.anthropic.claude-opus-4-7.json` | usage shape + pricing provider `anthropic` | `eu.anthropic.claude-haiku-4-5-20251001-v1:0`, `eu.anthropic.claude-opus-4-5-20251101-v1:0`, `eu.anthropic.claude-opus-4-6-v1`, `eu.anthropic.claude-opus-4-7`, `eu.anthropic.claude-sonnet-4-5-20250929-v1:0`, `eu.anthropic.claude-sonnet-4-6` |
| `eu.mistral.pixtral-large-2502-v1_0.json` | usage shape + pricing provider `mistral` | `eu.mistral.pixtral-large-2502-v1:0`, `mistral.devstral-2-123b`, `mistral.magistral-small-2509`, `mistral.ministral-3-14b-instruct`, `mistral.ministral-3-3b-instruct`, `mistral.ministral-3-8b-instruct`, `mistral.mistral-7b-instruct-v0:2`, `mistral.mistral-large-2402-v1:0`, `mistral.mixtral-8x7b-instruct-v0:1`, `mistral.voxtral-mini-3b-2507`, `mistral.voxtral-small-24b-2507` |
| `google.gemma-3-12b-it.json` | usage shape + pricing provider `google` | `google.gemma-3-12b-it`, `google.gemma-3-27b-it`, `google.gemma-3-4b-it` |
| `minimax.minimax-m2.1.json` | usage shape + pricing provider `minimax` | `minimax.minimax-m2.1`, `minimax.minimax-m2.5`, `minimax.minimax-m2` |
| `nvidia.nemotron-nano-12b-v2.json` | usage shape + pricing provider `nvidia` | `nvidia.nemotron-nano-12b-v2`, `nvidia.nemotron-nano-3-30b`, `nvidia.nemotron-nano-9b-v2` |
| `openai.gpt-oss-20b-1_0.json` | usage shape + pricing provider `openai` | `openai.gpt-oss-120b-1:0`, `openai.gpt-oss-20b-1:0`, `openai.gpt-oss-safeguard-120b`, `openai.gpt-oss-safeguard-20b` |
| `qwen.qwen3-32b-v1_0.json` | usage shape + pricing provider `qwen` | `qwen.qwen3-32b-v1:0`, `qwen.qwen3-coder-30b-a3b-v1:0`, `qwen.qwen3-next-80b-a3b`, `qwen.qwen3-vl-235b-a22b` |
| `zai.glm-4.7-flash.json` | usage shape + pricing provider `zai` | `zai.glm-4.7-flash` |
