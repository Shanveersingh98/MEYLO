import sqlite3

conn = sqlite3.connect('data/companion_project.db')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
counts = {t: c.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}
print(counts)
