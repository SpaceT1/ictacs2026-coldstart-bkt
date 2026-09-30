"""Step 2: clean the data (original problems, >=300 learners per skill)."""
import pandas as pd
df = pd.read_csv("data/as.csv", encoding="latin1", low_memory=False)
df = df[df.original == 1].dropna(subset=["skill_name"])
df = df.drop_duplicates(subset=["order_id", "skill_name"])
df = df[["order_id", "user_id", "skill_name", "correct"]]
df["correct"] = (df.correct >= 1).astype(int)
stu = df.groupby("skill_name").user_id.nunique()
df = df[df.skill_name.isin(stu[stu >= 300].index)].sort_values(["skill_name", "user_id", "order_id"])
df.to_csv("data/as_clean.csv", index=False)
print("rows", len(df), "learners", df.user_id.nunique(), "skills", df.skill_name.nunique())
