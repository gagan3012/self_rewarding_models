from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM,PreTrainedTokenizerFast, PreTrainedTokenizer, PreTrainedModel
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from tqdm import tqdm
import torch
import gc
import re
from datasets import load_dataset, Dataset
from fire import Fire
import pandas as pd
import random

def prompt_format(prompt, response, tokenizer):
    DEFAULT_LLM_AS_JUDGE_PROMPT = f"""
Review the user’s question and the corresponding response using the additive 5-point
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
User: {prompt}
<response>{response}</response>
After examining the user’s instruction and the response:
- Make sure to give a score between 1 and 10.
- Conclude with the score using the format: “Score: <total points>”
"""
    chat = [
    {"role": "user", "content": DEFAULT_LLM_AS_JUDGE_PROMPT},
    ]

    return tokenizer.apply_chat_template(chat, tokenize=False)

DEFAULT_REWARD_REGEX_TEMPLATE = """
Score: {{ reward }}
"""

def load_model(model_name, seed=42):
    llm = LLM(model=model_name, tokenizer=model_name, tensor_parallel_size=torch.cuda.device_count(), seed=seed, max_model_len=2048)
    return llm

def rank_responses(llm, responses, prompt, tokenizer):
    """
    Rank a list of responses based on a prompt
    """
    prompt_list = prompt_format(prompt, responses, tokenizer)
    sampling_params = SamplingParams(max_tokens=1024, temperature=1)
    outputs = llm.generate(prompt_list, sampling_params)

    # print([output.outputs[0].text.strip() for output in outputs][0])

    # Extract the reward from the response
    rewards = []
    for output in outputs:
        response = output.outputs[0].text.strip()
        reward = re.findall(r"Score: (\d)", response)[0] #or re.findall(r"Score: (\d\d)", response) or re.findall(r"Total score: (\d)", response)
        rewards.append(float(reward))

    # destroy_model_parallel()
    # gc.collect()
    # torch.cuda.empty_cache()
    return rewards


def chat_template(prompt, tokenizer):
    chat = [
    {"role": "user", "content": prompt},
    ]
    data = tokenizer.apply_chat_template(chat, tokenize=False)
    return data

def generate_responses(llm, prompt, tokenizer, seed=42):
    data = chat_template(prompt, tokenizer)
    sampling_params = SamplingParams(max_tokens=256, temperature=1)
    outputs = llm.generate(data, sampling_params)    
    return [output.outputs[0].text.strip() for output in outputs][0]


def main(model_name, seed=42, index=2):
    # Load the dataset
    dataset = load_dataset("/lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/ultrafeedback_binarized", 
                           cache_dir="/lustre07/scratch/gagan30/arocr/cache", 
                           split="train_gen", 
                           num_proc=4).shard(num_shards=5, index=index)
    
    # print(dataset)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Dictionary to store all responses for each seed
    df = []

    # Load the model
    llm = load_model(model_name, seed=seed)

    # Generate responses for each prompt in the dataset
    for prompt in tqdm(dataset['prompt']):
        responses = generate_responses(llm, prompt, tokenizer)

        # Rank the responses
        rewards = rank_responses(llm, responses, prompt, tokenizer)[0]

        # Append responses and their rewards to the DataFrame
        # for response, reward in zip(responses, rewards):
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": responses}
        ]
        df.append({'Seed':seed,'prompt': prompt, 'messages': chat, 'Reward': rewards})


    df = pd.DataFrame(df)
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache() 

    return df

def run(model_name,index):
    df = []
    for seed in range(1, 6):
        df.append(main(model_name, seed=seed, index=index))
    df = pd.concat(df)

    # print(df)

    rows = []

    # Iterate over unique prompts
    for prompt in df['prompt'].unique():
        prompt_df = df[df['prompt'] == prompt]
        prompt_df = prompt_df.reset_index(drop=True)
        # print(prompt_df)

        # Extract assistant messages and rewards
        all_responses = prompt_df['messages'].apply(lambda x: x[-1]['content']).tolist()
        all_rewards = prompt_df['Reward'].tolist()

        # Find the message with the highest reward
        chosen_index = prompt_df['Reward'].idxmax()
        chosen_message = prompt_df.iloc[chosen_index]['messages']

        # Get other messages and randomly select one for rejection
        other_messages = prompt_df.drop(chosen_index)['messages']
        rejected_message = random.choice(other_messages.to_list())

        # Append row to the list
        rows.append({
            'prompt': prompt, 
            'chosen': chosen_message, 
            'rejected': rejected_message, 
            'all_responses': all_responses, 
            'all_rewards': all_rewards
        })

    # Create final DataFrame from rows
    final_df = pd.DataFrame(rows)

    print(final_df)

    # Save the DataFrame to a JSON file

    ds = Dataset.from_pandas(final_df)

    model_name = model_name.split("/")[-1]

    ds.to_parquet(f"/lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/ultrafeedback_binarized/{model_name}.parquet")

if __name__ == "__main__":
    Fire(run)