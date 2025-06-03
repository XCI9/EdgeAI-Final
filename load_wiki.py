from datasets import load_dataset
import pandas as pd

dataset_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dataset_valid = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

text_train = "\n\n".join(dataset_train["text"])
text_valid = "\n\n".join(dataset_valid["text"])
merged_text = text_train + "\n\n" + text_valid

df = pd.DataFrame({"text": [merged_text]})
df.to_parquet("wikitext2_train_valid.parquet")

print("已儲存為 wikitext2_train_valid.parquet")
