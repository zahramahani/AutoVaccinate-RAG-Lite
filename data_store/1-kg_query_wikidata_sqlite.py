import sqlite3
import os

# === Path Configuration ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WIKIDATA_DIR = os.path.join(BASE_DIR, "data", "wikidata", "wikidata5m_transductive")
ALIAS_DIR = os.path.join(BASE_DIR, "data", "wikidata", "wikidata5m_alias")
SQLITE_DB = os.path.join(BASE_DIR, "data_store", "kg_wikidata5m.db")

def build_wikidata_sqlite():
    os.makedirs(os.path.dirname(SQLITE_DB), exist_ok=True)
    con = sqlite3.connect(SQLITE_DB)
    cur = con.cursor()

    # --- Create triples table ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS triples (
            subject TEXT,
            predicate TEXT,
            object TEXT
        )
    """)

    # --- Load triplets ---
    files = [
        os.path.join(WIKIDATA_DIR, "wikidata5m_transductive_train.txt"),
        os.path.join(WIKIDATA_DIR, "wikidata5m_transductive_valid.txt"),
        os.path.join(WIKIDATA_DIR, "wikidata5m_transductive_test.txt"),
    ]

    for path in files:
        if not os.path.exists(path):
            print(f"❌ Missing file: {path}")
            continue

        print(f"✅ Loading triples from {os.path.basename(path)} ...")
        with open(path, "r", encoding="utf-8") as f:
            rows = [tuple(line.strip().split("\t")) for line in f if line.strip()]
            cur.executemany("INSERT INTO triples (subject, predicate, object) VALUES (?, ?, ?)", rows)
            con.commit()
            print(f"   → Inserted {len(rows)} triples")

    # --- Create alias table ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS aliases (
            id TEXT,
            alias TEXT
        )
    """)

    alias_files = [
        os.path.join(ALIAS_DIR, "wikidata5m_entity.txt"),
        os.path.join(ALIAS_DIR, "wikidata5m_relation.txt"),
    ]

    for alias_path in alias_files:
        if not os.path.exists(alias_path):
            print(f"❌ Missing alias file: {alias_path}")
            continue

        print(f"✅ Loading aliases from {os.path.basename(alias_path)} ...")
        with open(alias_path, "r", encoding="utf-8") as f:
            alias_rows = []
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > 1:
                    entity_id = parts[0]
                    for alias in parts[1:]:
                        alias_rows.append((entity_id, alias))
            cur.executemany("INSERT INTO aliases (id, alias) VALUES (?, ?)", alias_rows)
            con.commit()
            print(f"   → Inserted {len(alias_rows)} aliases")

    print("✅ Wikidata5m SQLite database built successfully.")
    con.close()


if __name__ == "__main__":
    build_wikidata_sqlite()
