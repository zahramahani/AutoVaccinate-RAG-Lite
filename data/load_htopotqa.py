# from datasets import load_dataset

# # Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
print(ds)
print(ds['validation'][0])
subset = ds['validation'].select(range(1000))
subset.save_to_disk("./hotpotqa_1k")