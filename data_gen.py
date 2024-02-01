from datasets import load_dataset
import pprint

# dataset_bagel = load_dataset("parquet",data_files ="/lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/bagel-v0.3/bagel-clean-v0.3.parquet",
#                             cache_dir="/lustre07/scratch/gagan30/arocr/cache",
#                             split="train",
#                             num_proc=4)

# print(dataset_bagel)

# print(dataset_bagel[0])

# dataset_bagel = dataset_bagel.shuffle(seed=42)

# dataset_bagel = dataset_bagel.shard(num_shards=100, index=0)

# dataset_bagel = dataset_bagel.rename_column("conversations", "messages")

# print(dataset_bagel)

dataset = load_dataset("/lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/ultrafeedback_binarized", 
                           cache_dir="/lustre07/scratch/gagan30/arocr/cache", 
                           split="train_sft", 
                           num_proc=4).shard(num_shards=5, index=0)

print(dataset)

def setup_eft(sample):
    DEFAULT_LLM_AS_JUDGE_PROMPT = f"""Review the user’s question and the corresponding response using the additive 5-point
scoring system described below. Remember to assess from the AI Assistant perspective, utilizing web search knowledge as
necessary. To evaluate the response in alignment with this additive scoring model, we’ll
systematically attribute points based on the outlined criteria.
Points are accumulated based on the satisfaction of each
criterion:
- Add 1 point if the response is relevant and provides some information related to
the user’s inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s question,
but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user’s question in a
useful way, regardless of whether it seems to have been written by an AI Assistant or if it
has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective,
addressing the user’s question directly and comprehensively, and is well-organized and
helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s question
by an AI Assistant, without extraneous information, reflecting expert knowledge, and
demonstrating a high-quality, engaging, and insightful answer.
User: {sample['chosen'][0]['content'] if sample['chosen'][0]['role'] == 'user' else ''}
<response>{sample['chosen'][1]['content'] if sample['chosen'][1]['role'] == 'assistant' else ''}</response>
After examining the user’s instruction and the response:
- Make sure to give a score between 1 and 10.
- Conclude with the score using the format: “Score: <total points>”
"""
    sample['messages'] = [
        {"role": "user", "content": DEFAULT_LLM_AS_JUDGE_PROMPT},
        {"role": "assistant", "content": f"Score: {int(sample['score_chosen'])}"}
    ]
    return sample


def alternate_messages(sample):
    original_messages = sample.get('messages', [])
    eft_messages = sample.get('messages_eft', [])
    
    # Alternating between original and eft messages
    alternated_messages = []
    for orig, eft in zip(original_messages, eft_messages):
        alternated_messages.append(orig)
        alternated_messages.append(eft)

    # Handle the case where one list is longer than the other
    longest_list = original_messages if len(original_messages) > len(eft_messages) else eft_messages
    alternated_messages.extend(longest_list[len(alternated_messages)//2:])

    sample['messages'] = alternated_messages
    return sample


dataset = dataset.map(setup_eft)

pp = pprint.PrettyPrinter(indent=4,width=200, compact=True)
pp.pprint(dataset[0])

from datasets import concatenate_datasets

dataset2 = load_dataset("/lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/ultrafeedback_binarized", 
                           cache_dir="/lustre07/scratch/gagan30/arocr/cache", 
                           split="train_sft", 
                           num_proc=4).shard(num_shards=5, index=1)

ds = concatenate_datasets([dataset2, dataset])

ds = ds.shuffle(seed=42)

print(ds)

print(ds[0])

ds.to_parquet("/lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/ultrafeedback_binarized/train_srwm.parquet")