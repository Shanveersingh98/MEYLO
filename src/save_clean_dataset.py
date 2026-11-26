from datasets import load_dataset

OUTPUT = "D:/DEEP-LEARNING-PROJECT/data/processed/oasst1_en"

print("Loading OASST1...")
ds = load_dataset("OpenAssistant/oasst1")

print("Cleaning dataset...")
clean = ds["train"].filter(
    lambda x: x["lang"] == "en" and x["deleted"] == False
)

print(f"Clean messages: {len(clean)}")

print("Saving cleaned dataset...")
clean.save_to_disk(OUTPUT)

print("Done!")
print(f"Saved to: {OUTPUT}")