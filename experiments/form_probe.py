import os, time
from playwright.sync_api import sync_playwright
S="/datasets/nmahomed/vwa-adv/visualwebarena/.auth/reddit_state.json"
URL=os.environ.get("REDDIT","http://localhost:8080")+"/f/movies/86107"
with sync_playwright() as pw:
    b=pw.chromium.launch(headless=True); c=b.new_context(storage_state=S); p=c.new_page()
    p.goto(URL, wait_until="domcontentloaded", timeout=30000)
    for i,f in enumerate(p.locator("form").all()):
        print(f"form[{i}] action={f.get_attribute('action')} method={f.get_attribute('method')}")
        for t in f.locator("textarea").all():
            print(f"   textarea name={t.get_attribute('name')} id={t.get_attribute('id')}")
        for s in f.locator("button, input[type=submit]").all():
            print(f"   submit  text={(s.inner_text() or '').strip()[:30]!r} name={s.get_attribute('name')}")
    print("\nwho am I:", p.locator("a[href*='/user/']").first.inner_text() if p.locator("a[href*='/user/']").count() else "?")
    b.close()
