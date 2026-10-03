# archive/

Code kept for the record but not part of the pipeline that produced the
thesis results. Nothing here is imported by the runners, and nothing here is
hashed into a result row's code version.

| File | Why archived |
|---|---|
| `environments/som_resolver.py`, `experiments/som_resolver_live_check.py` | Set-of-marks resolver for the old generated-task VWA runner; the curated runner uses VisualWebArena's own action parsing. |
| `defenses/agent_sentry.py`, `tests/test_agent_sentry.py` | AgentSentry (counterfactual re-execution at each tool boundary). Not in the final defence set: it needs a dry-run loop the runners do not have. |
| `tests/test_icon_trace_prober.py` | Tests an earlier slope-based "trace prober" (`icon_trace_prober.py`, repo root) that ICON's Latent Space Trace Prober in `defenses/icon.py` replaced. |

Moved out of the hashed folders in P2a (3 Oct 2026), before the code freeze.
