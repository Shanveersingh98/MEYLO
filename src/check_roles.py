from generate_paths import clean, all_paths

print(f"Total conversation paths: {len(all_paths)}")

valid = 0
invalid = 0
examples = []

for path in all_paths:
    roles = [clean[i]["role"] for i in path]

    expected = ["prompter" if j % 2 == 0 else "assistant"
                for j in range(len(roles))]

    if roles == expected:
        valid += 1
    else:
        invalid += 1

        if len(examples) < 10:
            examples.append((path, roles))

print(f"Valid alternating paths: {valid}")
print(f"Invalid paths: {invalid}")

if examples:
    print("\nExamples of invalid paths:")

    for path, roles in examples:
        print("\n" + "=" * 70)

        for index, role in zip(path, roles):
            print(f"{index} | {role.upper()} | {clean[index]['text'][:150]}")