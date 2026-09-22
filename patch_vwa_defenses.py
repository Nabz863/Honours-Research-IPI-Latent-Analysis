"""
patch_vwa_defenses.py -- adds defence passes to the curated VWA-Adv driver.

Run once from the repo root:  python3 patch_vwa_defenses.py

What it adds
------------
1. `defense_names` parameter on run_curated_matrix, plus --defense_names CLI.
2. A payload cache populated during the attack loop and REUSED by the
   defended passes. Without this, GCG regenerates its ~150s/task payload
   for every defence -- 150s x 40 tasks x 3 defences is ~5 wasted hours,
   and the payload does not depend on the defence at all. This mirrors
   run_attack_matrix.py, which explicitly reuses payloads across
   defended passes for the same reason.
3. Two passes per defence, matching AgentDojo's structure exactly:
       defended_clean    -- defence on, no injection. This is the
                            defence's OWN footprint, and it is the
                            control analysis/rq2_geometry.py needs. RQ2
                            cannot be answered without it: you cannot
                            tell "moved somewhere new" from "moved the
                            way this defence always moves things".
       defended_attacked -- defence on, injection present.
   sample_id conventions match AgentDojo's so rq2_geometry.py runs on
   the output unchanged:
       defended_{defense}_{task}
       {attack}_defended_{defense}_{task}
"""

from pathlib import Path

p = Path("experiments/run_vwa_adv_curated_matrix.py")
s = p.read_text()

# --- 1. signature -----------------------------------------------------
old = """    exp_data: Path, site: str = "reddit",
    max_tasks: Optional[int] = None, max_steps: int = 15,
) -> None:"""
new = """    exp_data: Path, site: str = "reddit",
    max_tasks: Optional[int] = None, max_steps: int = 15,
    defense_names: Optional[List[str]] = None,
) -> None:"""
assert s.count(old) == 1, "anchor 1: signature"
s = s.replace(old, new)

# --- 2. payload cache, populated in the attack loop -------------------
old = """    extractor = ActivationExtractor(model, target_layers=target_layers, token_aggregation="last")
    registry = _build_registry(model, processor)"""
new = """    extractor = ActivationExtractor(model, target_layers=target_layers, token_aggregation="last")
    registry = _build_registry(model, processor)

    # (attack_name, task_name) -> (adv_cap, adv_img, meta). Populated by
    # the undefended attack loop and reused by every defended pass. The
    # adversarial payload is a function of the attack and the task, NOT
    # of the defence, so regenerating it per defence is pure waste --
    # and for GCG that waste is ~150s per task per defence.
    _payloads: Dict[tuple, tuple] = {}

    defense_registry = {}
    if defense_names:
        from experiments.run_attack_matrix import _build_defense_registry
        defense_registry = _build_defense_registry()
        for d in defense_names:
            if d not in defense_registry:
                raise KeyError(f"Unknown defense {d!r} -- have {sorted(defense_registry)}")"""
assert s.count(old) == 1, "anchor 2: registry"
s = s.replace(old, new)

# --- 3. store payloads before the episode runs ------------------------
old = """                vwa_task = VWAAdvTask(config_file=task.adv_config, task_id=task.name, intent=task.intent)"""
new = """                if attack_name != "clean":
                    _payloads[(attack_name, task.name)] = (adv_cap, adv_img, meta)

                vwa_task = VWAAdvTask(config_file=task.adv_config, task_id=task.name, intent=task.intent)"""
assert s.count(old) == 1, "anchor 3: payload store"
s = s.replace(old, new)

# --- 4. defended passes, inserted before env.close() ------------------
old = """                logger.info("  done: %s (ASR=%s, UA=%s, gen=%s, ep=%.1fs)",
                            sid, asr, ua, f"{gen_s:.1f}s" if gen_s else "n/a", ep_s)
    finally:
        env.close()"""
new = '''                logger.info("  done: %s (ASR=%s, UA=%s, gen=%s, ep=%.1fs)",
                            sid, asr, ua, f"{gen_s:.1f}s" if gen_s else "n/a", ep_s)

        # ---------------- defended passes ----------------------------
        # Two per defence. defended_clean measures the defence's own
        # displacement with nothing to defend against; without it,
        # rq2_geometry.py has no control and the restoration-versus-
        # suppression question is unanswerable.
        for defense_name in (defense_names or []):
            defense = defense_registry[defense_name]

            logger.info("--- defended CLEAN: %s (%d tasks) ---",
                        defense_name, len(usable))
            for task in usable:
                sid = f"defended_{defense_name}_{task.name}"
                if done(sid):
                    logger.info("  skip %s", sid); continue
                vwa_task = VWAAdvTask(config_file=task.adv_config,
                                      task_id=task.name, intent=task.intent)
                t0 = time.perf_counter()
                with _cuda_memory_guard() as mem:
                    out, err = _run_episode_safely(
                        run_vwa_adv_episode, model, processor, extractor,
                        model_id, vwa_task, env,
                        instruction_path=str(VWA_ADV_REPO / "agent/prompts/jsons/p_som_cot_id_actree_3s.json"),
                        max_steps=max_steps, adv_url2caption={}, adv_url2image={},
                        defense=defense,
                    )
                ep_s = time.perf_counter() - t0
                traj = out.get("trajectory")
                asr = score(task.adv_config, traj) if traj else None
                ua = score(task.clean_config, traj) if (traj and task.ua_scorable) else None
                torch.save(out.get("step_activations", []), output_dir / f"{sid}.pt")
                record(sid, {
                    "sample_id": sid, "condition": "defended_clean",
                    "attack_name": "clean", "defense_name": defense_name,
                    "task_id": task.name, "base_task": task.base, "site": site,
                    "intent": task.intent, "adversarial_goal": task.adversarial_goal,
                    "strict_asr": getattr(task, "strict_asr", False),
                    "asr_success": asr, "utility_success": ua,
                    "ua_note": None if task.ua_scorable else "clean evaluator targets a different intent -- UA not scorable",
                    "num_steps": out.get("num_steps"), "attack_generation_time_s": None,
                    "episode_time_s": ep_s, "peak_cuda_memory_gb": mem["peak_gb"],
                    "error": err,
                })
                logger.info("  done: %s (ASR=%s, UA=%s, ep=%.1fs)", sid, asr, ua, ep_s)

            for attack_name in attack_names:
                logger.info("--- defended %s: %s (%d tasks) ---",
                            attack_name, defense_name, len(usable))
                for task in usable:
                    sid = f"{attack_name}_defended_{defense_name}_{task.name}"
                    if done(sid):
                        logger.info("  skip %s", sid); continue
                    cached = _payloads.get((attack_name, task.name))
                    if cached is None:
                        # The undefended pass for this (attack, task) did
                        # not produce a payload -- generation failed, or
                        # this is a resumed run whose attack loop was
                        # skipped. Either way, defending nothing would
                        # silently produce a row labelled defended that
                        # is really a clean episode.
                        record(sid, {"sample_id": sid, "attack_name": attack_name,
                                     "defense_name": defense_name, "task_id": task.name,
                                     "error": "no cached payload from the undefended pass"})
                        continue
                    adv_cap, adv_img, meta = cached
                    vwa_task = VWAAdvTask(config_file=task.adv_config,
                                          task_id=task.name, intent=task.intent)
                    t0 = time.perf_counter()
                    with _cuda_memory_guard() as mem:
                        out, err = _run_episode_safely(
                            run_vwa_adv_episode, model, processor, extractor,
                            model_id, vwa_task, env,
                            instruction_path=str(VWA_ADV_REPO / "agent/prompts/jsons/p_som_cot_id_actree_3s.json"),
                            max_steps=max_steps, adv_url2caption=adv_cap,
                            adv_url2image=adv_img, defense=defense,
                        )
                    ep_s = time.perf_counter() - t0
                    traj = out.get("trajectory")
                    asr = out.get("success")
                    ua = score(task.clean_config, traj) if (traj and task.ua_scorable) else None
                    torch.save(out.get("step_activations", []), output_dir / f"{sid}.pt")
                    record(sid, {
                        "sample_id": sid, "condition": "defended_attacked",
                        "attack_name": attack_name, "defense_name": defense_name,
                        "task_id": task.name, "base_task": task.base, "site": site,
                        "intent": task.intent, "adversarial_goal": task.adversarial_goal,
                        "strict_asr": getattr(task, "strict_asr", False),
                        "asr_success": asr, "utility_success": ua,
                        "ua_note": None if task.ua_scorable else "clean evaluator targets a different intent -- UA not scorable",
                        "num_steps": out.get("num_steps"),
                        "attack_generation_time_s": 0.0,  # reused, not regenerated
                        "episode_time_s": ep_s, "peak_cuda_memory_gb": mem["peak_gb"],
                        "error": err, **meta,
                    })
                    logger.info("  done: %s (ASR=%s, UA=%s, ep=%.1fs)", sid, asr, ua, ep_s)
    finally:
        env.close()'''
assert s.count(old) == 1, "anchor 4: defended passes"
s = s.replace(old, new)

# --- 5. CLI -----------------------------------------------------------
old = '    ap.add_argument("--output_dir", type=str, default="vwa_adv_curated")'
new = ('    ap.add_argument("--defense_names", type=str, nargs="*", default=None,\n'
       '                    help="spotlighting_delimiting spotlighting_datamarking "\n'
       '                         "spotlighting_encoding; runs defended-clean AND "\n'
       '                         "defended-attacked passes per defence")\n'
       + old)
assert s.count(old) == 1, "anchor 5: cli"
s = s.replace(old, new)

# --- 6. pass it through from __main__ ---------------------------------
old = """                       Path(a.exp_data), a.site, a.max_tasks, a.max_steps)"""
new = """                       Path(a.exp_data), a.site, a.max_tasks, a.max_steps,
                       defense_names=a.defense_names)"""
assert s.count(old) == 1, "anchor 6: call site"
s = s.replace(old, new)

p.write_text(s)
print("VWA-Adv driver: defence passes added (payload-reusing)")
