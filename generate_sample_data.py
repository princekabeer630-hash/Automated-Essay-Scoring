import pandas as pd
import random

random.seed(42)

topics = [
    "Technology has changed the way we live and communicate with each other. It has made life easier but also brought new challenges that society must deal with carefully.",
    "Education is the key to success in life. Students who study hard and stay focused usually achieve their goals and build a better future for themselves.",
    "Environmental pollution is a serious problem affecting our planet. Governments and individuals must work together to reduce waste and protect natural resources.",
    "Sports play an important role in keeping people healthy. Regular physical activity improves both physical and mental well being of a person.",
    "Reading books helps expand our knowledge and imagination. It is one of the best habits a person can develop from an early age.",
]

low_quality_extra = " i think this good because it help alot and also make sense to me and everyone agree with this idea i has write."
high_quality_extra = " Furthermore, this demonstrates a nuanced understanding of the subject, supported by coherent reasoning and well-structured arguments that strengthen the overall argument."

rows = []
for i in range(300):
    base = random.choice(topics)
    quality = random.choice(["low", "medium", "high"])

    if quality == "low":
        essay = base[:80] + low_quality_extra
        score = random.randint(1, 4)
    elif quality == "medium":
        essay = base
        score = random.randint(4, 7)
    else:
        essay = base + " " + base + high_quality_extra
        score = random.randint(7, 10)

    rows.append({"essay": essay, "score": score})

df = pd.DataFrame(rows)
df.to_csv("data/essays.csv", index=False)
print("Sample dataset bann gaya: data/essays.csv")
print(df.head())
