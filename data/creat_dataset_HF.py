from datasets import load_dataset
import pandas as pd

dataset = load_dataset("imdb" ,split="train")

df = pd.DataFrame({
    "text" : dataset["text"],
    "label" : ["positive" if l ==1 else "negetive" for l in dataset["label"]]
})

df.to_csv("data/imdb.csv" , index = False)
print(df.head(10))
print(df.__len__())
print("DONE!!")