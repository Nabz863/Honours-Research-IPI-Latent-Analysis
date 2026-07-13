"""
attacks package
----------------
One module per IPI attack method (Table 3.2 of the proposal). See
attacks/base.py for the shared interface every attack implements.

Design intent: a single implementation per attack TYPE, instantiated
and called many times across different (task, target_behavior, model)
combinations. The "hundreds of attack instances" reported in
benchmarking papers come from this combinatorial application over
existing task suites (VWA-Adv, AgentDojo) -- not from hand-writing a
separate script per instance.
"""
