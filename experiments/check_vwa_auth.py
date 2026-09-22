"""
experiments/check_vwa_auth.py
-----------------------------
Is the agent actually logged in to Postmill?

WHY THIS EXISTS
    Nearly every curated reddit task is "leave a comment" / "create a
    comment", which Postmill only permits when authenticated. On the
    strict-scored run, clean UA was 0/9 and 12/19 episodes terminated at
    exactly 4 steps -- which is _early_stop's parsing_failure threshold
    (3 unparseable actions) plus the stop action. A silently
    unauthenticated agent produces exactly that: it tries to comment,
    the control isn't there, its actions don't parse, the episode ends.

    run_vwa_adv_episode() does not renew the session. The benchmark's
    own run.py calls browser_env/auto_login.py before each task; the
    curated driver never does. This is flagged as risk #2 in
    run_one_vwa_adv_episode.py's docstring and has never been tested.

WHY THE STORED SESSION IS SUSPECT
    .auth/reddit_state.json was captured 2026-08-06 and holds:
        PHPSESSID   session cookie  -- server-side record
        REMEMBERME  expires 2027    -- may be signature-based
        _cookie_check
    Postmill runs under `apptainer --writable-tmpfs`, so its Postgres
    data is discarded on instance stop and rebuilt from the image each
    job. A server-side session from August cannot exist in a database
    created last night. Whether REMEMBERME still authenticates depends
    on whether Symfony's remember-me is signature-based (survives) or
    table-backed (does not).

    That is an empirical question, so this script answers it rather
    than reasoning about it.

WHAT IT CHECKS
    1. Postmill is reachable at $REDDIT.
    2. With the stored state loaded, is the browser authenticated?
       Detected by the presence of a logout control / absence of a login
       link, not by page title.
    3. Can it actually reach a comment form on a real post?
    4. Optionally, does a real comment submission persist? (--post)

Run this INSIDE the job serving Postmill. No GPU, no model.

    python3 experiments/check_vwa_auth.py
    python3 experiments/check_vwa_auth.py --post     # writes a comment
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

VWA_ADV_REPO = Path(os.environ.get(
    "VWA_ADV_REPO", "/datasets/nmahomed/vwa-adv/visualwebarena"))
STATE = VWA_ADV_REPO / ".auth" / "reddit_state.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("REDDIT", "http://localhost:8080"))
    ap.add_argument("--post_url", default=None,
                    help="a real post to test commenting on; defaults to a "
                         "curated task's own start_url")
    ap.add_argument("--username", default="MarvelsGrantMan136",
                    help="the account the evaluators check for comments")
    ap.add_argument("--post", action="store_true",
                    help="actually submit a comment and verify it persists")
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright

    post_url = a.post_url or f"{a.url.rstrip('/')}/f/movies/86107"

    print(f"REDDIT       : {a.url}")
    print(f"storage_state: {STATE}  exists={STATE.exists()}")
    if STATE.exists():
        d = json.load(open(STATE))
        print(f"  captured   : {time.ctime(STATE.stat().st_mtime)}")
        for c in d.get("cookies", []):
            e = c.get("expires", -1)
            when = "session" if e in (-1, None) else time.ctime(e)
            print(f"  cookie     : {c.get('name'):16s} {c.get('domain')}  expires={when}")

    ok = True
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)

        # --- anonymous control: what does logged-OUT look like? --------
        anon = browser.new_context()
        pa = anon.new_page()
        pa.goto(post_url, wait_until="domcontentloaded", timeout=30000)
        anon_login_links = pa.locator("a[href*='/login']").count()
        anon_forms = pa.locator("textarea").count()
        print(f"\n[anonymous] login links={anon_login_links}  textareas={anon_forms}")
        anon.close()

        # --- with the stored session -----------------------------------
        ctx = browser.new_context(storage_state=str(STATE) if STATE.exists() else None)
        p = ctx.new_page()
        try:
            p.goto(post_url, wait_until="domcontentloaded", timeout=30000)
        except Exception as e:
            print(f"FATAL: cannot reach {post_url} -- {type(e).__name__}: {e}")
            sys.exit(2)

        login_links = p.locator("a[href*='/login']").count()
        logout_links = p.locator("a[href*='/logout']").count()
        textareas = p.locator("textarea").count()
        body = p.content()
        name_present = a.username.lower() in body.lower()

        print(f"[stored   ] login links={login_links}  logout links={logout_links}  "
              f"textareas={textareas}  username_in_page={name_present}")

        authed = logout_links > 0 or (login_links == 0 and anon_login_links > 0)
        print(f"\nAUTHENTICATED: {authed}")
        if not authed:
            ok = False
            print("  -> The stored session does NOT authenticate against this")
            print("     instance. Every 'leave a comment' task is unattemptable,")
            print("     which explains UA=0 independent of model capability.")
            print("     Fix: call browser_env/auto_login.py before env.reset(),")
            print("     the way the benchmark's own run.py does.")

        if textareas == 0:
            ok = False
            print("  -> No comment form present on the post page.")

        # --- optional: prove a comment actually persists ---------------
        if a.post and authed and textareas:
            marker = f"authcheck-{int(time.time())}"
            try:
                # Target the comment form specifically. The first
                # textarea on the page is the comment box, but the first
                # submit button belongs to form[0], the SEARCH form --
                # clicking that submits a search and posts nothing, which
                # produced a false "comment did not persist" result.
                form = p.locator("form[action*='/-/comment']").first
                form.locator("textarea").first.fill(marker)
                form.locator("button[type='submit'], button:has-text('Post'), "
                             "input[type='submit']").first.click()
                p.wait_for_load_state("domcontentloaded", timeout=30000)
                p.goto(post_url, wait_until="domcontentloaded", timeout=30000)
                persisted = marker in p.content()
                print(f"\nCOMMENT PERSISTED: {persisted}  (marker {marker})")
                if not persisted:
                    ok = False
                    print("  -> Submission accepted but the comment is not on the")
                    print("     page. The evaluators read comments via")
                    print("     reddit_get_latest_comment_content_by_username, so")
                    print("     they would score False regardless of agent behaviour.")
            except Exception as e:
                ok = False
                print(f"\nCOMMENT POST FAILED: {type(e).__name__}: {e}")

        ctx.close()
        browser.close()

    print("\n" + ("PASS -- auth is not the problem" if ok else
                  "FAIL -- auth/commenting is broken; behavioural results "
                  "from this environment are not interpretable"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
