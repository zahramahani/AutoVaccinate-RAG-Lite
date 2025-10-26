# coverage_check.py
from linker import extract_entities, link_entity, normalize_label
import json

claims = [
    "Was Colin Kaepernick a quarterback for the 49ers?",
    "Who directed Moonraker?",
    "Did Marlene Dietrich star in Kismet?"
]

total = 0
covered = 0
for c in claims:
    ents = extract_entities(c)
    print("\nCLAIM:", c)
    for e in ents:
        total += 1
        qids = link_entity(e, topk=3)
        print("  ENT:", e, "->", qids)
        if qids:
            covered += 1

print(f"\nCoverage: {covered}/{total} entities covered")
