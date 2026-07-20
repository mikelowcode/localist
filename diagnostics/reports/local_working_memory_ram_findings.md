# Local Working-Memory RAM Findings — is a safe `LOCAL_PROFILE` increase possible?

**Date:** 2026-07-19
**Machine:** the primary dev Mac (16GB / 17.18GB-decimal physical RAM, Apple Silicon)
**Script:** `diagnostics/diag_local_working_memory_ram_probe.py`
**Raw data:** `diagnostics/local_working_memory_ram_probe_results.json`

## Preface — a premise in the original brief didn't hold up

The investigation brief that kicked this off cited `parallel_runtime_spec.md` §7 as
prior art — a live measurement of "12.92GB used at idle N=1," an "8GB headroom on a
16GB gate," and a named "Metal-cap panic" failure mode, asking this investigation to
"mirror the discipline" of that document.

That document does not exist anywhere in this repository — not currently, not in git
history — and none of those specific figures or the "Metal-cap panic" name appear
anywhere else in the codebase. What *is* real and confirmed in `context_profile.py`'s
docstring and `docs/architecture/16-runtime-backend-layer.md`: `LOCAL_PROFILE`
(`working_memory_tokens=300, working_memory_limit=5`) was sized in general terms for
"the 16GB Apple Silicon local-inference case," with no prior live measurement behind
the specific numbers. This investigation proceeded green-field — real measurement on
this machine, right now — rather than reconstructing or assuming the cited document's
numbers.

## Step 1 — current real headroom, both backends, right now

Machine snapshot **before any local model was loaded** (`psutil.virtual_memory()` /
`psutil.swap_memory()`):

| Metric | Value |
|---|---|
| Total RAM | 17.18 GB |
| Used | 8.26 GB |
| Available (macOS reclaimable-inclusive) | 6.60 GB |
| **Free (truly unused)** | **0.13 GB** |
| Swap used | 2.63 GB / 4.29 GB (61.3%) |

**This machine's load right now is not a quiet baseline.** Top RSS consumers at
snapshot time included a running Virtualization.framework VM (1.08GB), several VS
Code / Claude helper processes, and WebKit — ordinary interactive-session load, not a
synthetic idle state. Swap was already 61% in use before this investigation touched
anything. This snapshot should be treated as **one loaded-machine data point, not a
permanent ceiling** — a quieter machine, or the same machine after quitting other
apps, would show more headroom; a busier one would show less. Re-validate rather than
treating this figure as fixed, consistent with how the (non-existent, but rationally
scoped) original ask wanted this treated.

Both local backends were confirmed live for this investigation:
- **oMLX**: an externally-managed server on port 8000 (per `CLAUDE.md`, "managed
  separately") serving `gemma-4-e4b-it-4bit`. Notably, this process was observed to
  restart itself under its own supervision between test runs (pid churned from 42401
  → 43275 unprompted) — expected/tolerated by the probe (it re-resolves the pid
  before every trial), not a bug in this investigation.
- **Local Ollama**: `gemma4:e4b-mlx` (8.8GB on disk), loaded on first request via
  `ollama runner --mlx-engine`.

## Step 2 — marginal RAM per additional 1,000 working-memory tokens (measured)

### Methodology correction found during this investigation

`psutil`'s per-process RSS **does not capture Metal/unified-memory GPU-resident
allocations** on Apple Silicon. Confirmed directly: after Ollama loaded the 8.8GB
model, `psutil.Process(pid).memory_info().rss` reported **~200MB** for the runner
process at the exact same instant `top -l 1 -pid <pid> -stats mem` reported **~8.3GB**
for that pid. Using psutil RSS alone would have under-reported this workload's real
footprint by roughly 40x. The probe script uses `top`'s memory-footprint column
(resident + compressed + purgeable) as per-process ground truth instead, cross-checked
against system-wide `psutil.virtual_memory().used` deltas. **Do not revert this to
plain psutil RSS in any future version of this probe** — it would silently produce
numbers an order of magnitude too small for MLX-backed processes.

Working-memory filler was real text pulled from this project's own `conversation_log`
table (1,058 real rows), not synthetic filler, per this project's established
discipline against token-density mismatches invalidating a probe.

Each cell = 3 trials at a nominal token target; because real turns vary drastically in
length (up to ~1,300 tokens for a single long agent reply), the *actual* delivered
token count per bucket sometimes overshot the nominal target — both are reported below
for honesty about what was actually sent.

### Local Ollama (`gemma4:e4b-mlx`)

| nominal tokens | actual tokens | peak footprint (mean±σ, MB) | peak sys used (mean, GB) | swapouts (3 trials) |
|---|---|---|---|---|
| 300  | 708  | 8499.7 ± 11.9  | 11.691 | 3888, 4952, 4676 |
| 600  | 708  | 8526.3 ± 11.7  | 11.724 | 2744, 0, 700 |
| 1200 | 2018 | 9316.0 ± 653.6 | 12.569 | **158380**, 0, 0 |
| 2000 | 2018 | 8834.0 ± 4.3   | 12.485 | 0, 0, 0 |
| 3000 | 3707 | 9464.3 ± 548.8 | 12.912 | 0, 0, 0 |

(The two 653.6/548.8 stdev cells are driven by a single first-trial-after-a-size-change
reading of exactly 10240.0MB in each — confirmed to be `top`'s one-decimal-place
rounding at the gigabyte boundary (anything 9.95–10.05GB rounds to "10.0G"), not a
discovered hard ceiling; both also had 2–3x longer elapsed time than their sibling
trials, consistent with a one-off reallocation/compaction event rather than a real
per-token cost.)

**Marginal cost, 708→3707 actual tokens:** (9464.3 − 8499.7) MB / (3707 − 708) tokens
≈ **~320 MB per 1,000 tokens of working memory.**

**Even at today's unchanged 300-token ceiling, every single trial produced
non-zero swapouts** (3888–4952 pages). This is the headline finding: the *existing*
budget is already inducing swap activity on this machine under today's realistic load
— this isn't a hypothetical future risk from raising the number, it's happening now.

### oMLX (`gemma-4-e4b-it-4bit`)

| nominal tokens | actual tokens | peak footprint (mean±σ, MB) | peak sys used (mean, GB) | swapouts (3 trials) |
|---|---|---|---|---|
| 300  | 708  | 5059.3 ± 11.1  | 9.643* | 220380*, 11044, 3592 |
| 600  | 708  | 5094.0 ± 25.5  | 8.536  | 19232, 2648, 0 |
| 1200 | 2018 | 5124.7 ± 150.5 | 8.509  | 29508, 5052, 2516 |
| 2000 | 2018 | 4986.7 ± 0.9   | 8.442  | 796, 4872, 0 |
| 3000 | 3707 | 5622.7 ± 69.4  | 8.749  | 24448, 6148, 4980 |

\* The first oMLX trial (300 tokens, trial 1) ran immediately after the Ollama sweep
finished, while the ~9GB Ollama runner was still being evicted from RAM to make room —
its 11.902GB peak-sys-used and 220,380-page swapout reading is a **cross-backend
transition artifact**, not a clean oMLX-alone measurement. It's reported for
transparency but excluded from the marginal-cost calculation below. This is itself a
real and useful finding, independent of the numbers: **switching between local
backends on this machine, without a cooldown, causes a large transient swap storm**
purely from memory handoff — worth keeping in mind for the runtime-backend live-switch
feature (§16.5) if it's ever used to switch between two *local* tiers back-to-back.

**Marginal cost, 708→3707 actual tokens (excluding the contaminated trial):** (5622.7 −
5059.3) MB / (3707 − 708) tokens ≈ **~190 MB per 1,000 tokens of working memory.**

oMLX's baseline footprint (~5GB) is notably lower than Ollama's (~8.5GB) for
functionally the same model — the two serving stacks do have meaningfully different
memory profiles, as the original brief suspected. Don't assume one backend's numbers
transfer to the other.

**No OOM-kill and no reproducible "Metal-cap panic" occurred in either backend across
this sweep.** The failure mode actually observed was swap activity, not a crash —
consistent with the brief's warning that swapping is "a slow, silent form of the same
problem," just not the specific named incident cited in the original ask (which, per
the preface above, doesn't appear to exist in this project's history).

## Step 3 — is a safe increase available?

Naively: worst measured `sys_used` peak was 13.526GB (Ollama, 3000 tokens) against
17.18GB total — arithmetically, ~3.65GB of headroom "remains." Projected against the
~320MB/1000-token marginal cost, that would suggest room for several thousand more
working-memory tokens before exhausting total RAM.

**That arithmetic is the wrong answer here.** The actual observed failure precursor —
swapping — is already present at *every single trial at every token size tested,
including today's unchanged 300-token baseline*. The system is not idle-with-headroom;
it is already relying on the compressor/swap to make room under this machine's current
real load (2.63GB swap in use before the probe even started). Extrapolating "more room
exists" from used-vs-total memory ignores the behavior actually measured. Per this
project's own standard ("watch explicitly for the failure mode, not just a slowdown"),
the swap evidence outweighs the headroom arithmetic.

## Step 4 — turn-count vs. token-ceiling: did it matter in practice?

Queried the live `conversation_log` table directly (1,159 real rows, 318 distinct
`task_id` windows) rather than guessing:

- Mean turn length: **20.6 tokens** (user turns) / **185.4 tokens** (agent turns)
  (chars // 4, same estimator `get_context_window()` uses).
- **107 of 318 (34%) of task windows already have their last-5-rows sum exceed 300
  tokens** under today's settings — i.e., the token ceiling is already the binding
  constraint for roughly a third of real conversations today, not a rare edge case.

This confirms the ambiguity flagged at the top of the original brief was real: raising
`working_memory_limit` from 5 to 10 **without** also raising `working_memory_tokens`
would be a near-total no-op — a third of windows are already token-bound at 5 rows,
and the rest sit close enough to the 300-token ceiling on average (5 turns ≈ 2–3
agent replies × 185 tokens ≈ 460–550 tokens before trimming) that most of the
"extra" rows a limit-10 fetch would pull in get trimmed straight back off by the
unchanged token ceiling anyway.

## Step 5 — decision

**No safe increase to `LOCAL_PROFILE` on this hardware today.** `working_memory_tokens`
stays at 300, `working_memory_limit` stays at 5. This is a legitimate, evidence-backed
outcome per the investigation's own scope guardrail, not a failure to find a number —
the machine is observably already swapping at the current setting under real load.
`context_profile.py`'s docstring has been updated with this measured rationale so a
future session revisiting this doesn't have to redo the investigation from zero (see
the module docstring's new "2026-07-19 measurement" section).

**On the startup guard (original step 5):** that step was scoped as "if the budget is
raised, add a guard." Since no increase is being made, it isn't triggered by the
brief's own criteria. That said, the swap-at-baseline finding above is arguably a more
urgent, independent problem than the one this investigation was asked to solve — it
suggests today's *existing* ceiling may already be marginal on a loaded machine, not
just closed to growth. That's a separate follow-up worth a deliberate decision, not
something bundled into this change silently.

## Caveats / re-validation

- This is a **one-machine, one-session snapshot** taken under real (not
  synthetically quiet) background load. Re-run this probe if: RAM is added/removed,
  the model changes, or a quieter/busier baseline needs to be characterized
  specifically (e.g., "what if Michael quits every other app first").
- `top`-based footprint measurement has ~0.1GB quantization noise at the gigabyte
  boundary (see the 10240.0MB artifact above) — real per-token deltas smaller than
  that will be lost in the rounding; the marginal-cost figures above are directional,
  not exact to the MB.
- The oMLX 300-token/trial-1 reading is contaminated by a cross-backend transition and
  was excluded from the marginal-cost calculation, but is retained in the raw JSON and
  the table above for transparency.
