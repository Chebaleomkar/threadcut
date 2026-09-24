"""Prove every task is well-formed: tests fail as shipped and pass with the intended fixes.

Usage: python tasks/check_solvable.py   (works on a temp copy; the tasks themselves are untouched)
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

FIXES = {
    "inventory/inventory/models.py": [("self.price_cents + self.qty", "self.price_cents * self.qty")],
    "inventory/inventory/store.py": [("if item.qty < 0:", "if item.qty == 0:"), ("i.qty < threshold", "i.qty <= threshold")],
    "inventory/inventory/report.py": [("{cents / 100}", "{cents / 100:.2f}"), ("key=lambda i: i.sku", "key=lambda i: i.name")],
    "textstats/textstats/tokenize.py": [("[A-Za-z']+", "[A-Za-z0-9']+"), ("WORD.findall(text)", "WORD.findall(text.lower())")],
    "textstats/textstats/counts.py": [("key=lambda kv: (kv[1], kv[0])", "key=lambda kv: (-kv[1], kv[0])")],
    "textstats/textstats/summary.py": [("unique = len(words)", "unique = len(set(words))"),
                                       ("/ len(words)", "/ len(words) if words else 0")],
    "ledger/ledger/account.py": [("if amount < 0:", "if amount <= 0:"), ("< self.overdraft_limit", "< -self.overdraft_limit")],
    "ledger/ledger/transfer.py": [("round(amount * rate_bp / 10000)", "(amount * rate_bp + 5000) // 10000"),
                                  ("    dst.deposit(amount)\n    src.withdraw(amount + fee(amount, rate_bp))",
                                   "    src.withdraw(amount + fee(amount, rate_bp))\n    dst.deposit(amount)")],
    "csvpipe/csvpipe/parse.py": [(
        '    lines = text.splitlines()\n    header = [h.strip() for h in lines[0].split(",")]\n    rows = []\n'
        '    for line in lines[1:]:\n        fields = [f.strip() for f in line.split(",")]',
        "    import csv\n    lines = [l for l in text.splitlines() if l.strip()]\n"
        "    recs = list(csv.reader(lines, skipinitialspace=True))\n    header = [h.strip() for h in recs[0]]\n"
        "    rows = []\n    for rec in recs[1:]:\n        fields = [f.strip() for f in rec]")],
    "csvpipe/csvpipe/clean.py": [("if qty < 0:", "if qty <= 0:"), (".capitalize()", ".title()")],
    "csvpipe/csvpipe/aggregate.py": [('r["qty"] + r["price"]', 'r["qty"] * r["price"]'), ("totals.items()}", "sorted(totals.items())}")],
    "life/life/grid.py": [("if (r + dr, c + dc) in self.alive:",
                           "if (dr or dc) and ((r + dr) % self.rows, (c + dc) % self.cols) in self.alive:")],
    "life/life/render.py": [("(c, r) in g.alive", "(r, c) in g.alive")],
    "schedule/schedule/timeparse.py": [('    if ampm == "pm":\n        h += 12',
                                        '    if ampm == "pm" and h != 12:\n        h += 12\n'
                                        '    if ampm == "am" and h == 12:\n        h = 0')],
    "schedule/schedule/meetings.py": [("a.start <= b.end and b.start <= a.end", "a.start < b.end and b.start < a.end"),
                                      ("(b.title, a.title)", "(a.title, b.title)")],
}


def pytest_ok(d):
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=d,
                          capture_output=True).returncode == 0


def main():
    here = pathlib.Path(__file__).parent
    tasks = sorted(p.name for p in here.iterdir() if (p / "prompt.txt").exists())
    with tempfile.TemporaryDirectory() as tmp:
        for t in tasks:
            shutil.copytree(here / t, pathlib.Path(tmp) / t)
        for f, subs in FIXES.items():
            p = pathlib.Path(tmp) / f
            s = p.read_text()
            for a, b in subs:
                assert a in s, (f, a)
                s = s.replace(a, b)
            p.write_text(s)
        for t in tasks:
            fails = not pytest_ok(here / t)
            fixed = pytest_ok(pathlib.Path(tmp) / t)
            print(f"{t:10s} fails as shipped: {fails}  passes when fixed: {fixed}")
            assert fails and fixed, t


if __name__ == "__main__":
    main()
