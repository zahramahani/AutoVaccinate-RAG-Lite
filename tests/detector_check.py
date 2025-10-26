from detectors import extract_candidate_assertions
examples = [
    "Colin Kaepernick was a quarterback for the 49ers.",
    "Marlene Dietrich starred in Kismet.",
    "Moonraker was directed by Lewis Gilbert.",
    "Barack Obama was born in Hawaii."
]

for t in examples:
    print(t)
    print(extract_candidate_assertions(t))
    print("----")
