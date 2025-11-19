import argparse
import base64
import io
import json
import os
import re
import sys
import time
import collections
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Add parent directory to sys.path to allow imports from sibling packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml

from datasets import Dataset
from openai import OpenAI
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

from custom_rewards.lmms_lab_recipe import compute_score, extract_anwser_tag

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts and nothing else: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)

MODEL_VERSION = os.environ.get("MODEL_VERSION", "qwen3-vl")


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_parquet_dataset(parquet_path: str) -> "Dataset":
    return Dataset.from_parquet(parquet_path)


def encode_local_image_to_data_url(file_path: str) -> str:
    with Image.open(file_path) as img:
        img = img.convert("RGB")
        with io.BytesIO() as buf:
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
    return encode_image_to_data_url(image_bytes, "image/png")


def encode_image_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def get_client() -> OpenAI:
    client = OpenAI(
        base_url=os.environ.get("OAI_BASE_URL"),
        api_key=os.environ.get("OAI_API_KEY"),
        timeout=6000,
    )
    return client

def print_oai_messages(messages: List[Dict[str, Any]]):
    # Print the messages in a readable format, exclude the image_url
    for message in messages:
        role = message["role"]
        content = message["content"]
        for item in content:
            if item["type"] == "text":
                print(f"{role}: {item['text']}")
            elif item["type"] == "image_url":
                print(f"{role}: <image>")

def handle_question(question: str) -> str:
    question = question.replace("<image>", "")
    question = question.replace("Context: N/A", "")
    question = question.replace("Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.", "")
    question = question.replace("Answer the question using a single word or phrase.", "")
    return question.strip()

def handle_output_text(output_text: str) -> str:
    # Some model manually add <think> tag in chat template, so we need to handle this
    if "</think>" in output_text and "<think>" not in output_text:
        output_text = "<think>" + output_text
    if "</answer>" in output_text and "<answer>" in output_text and "<think>" not in output_text:
        content_before_answer, content_after_answer = output_text.rsplit("<answer>", 1)
        content_before_answer = content_before_answer.strip()
        content_after_answer = content_after_answer.strip()
        output_text = "<think>" + content_before_answer + "</think><answer>" + content_after_answer
    output_text = output_text.replace("<tool_call>", "")
    # Search for <think> </think> and <answer> </answer> tags, and return the text between them, using regex
    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
    if think_match:
        think = think_match.group(1)
    else:
        think = None
    if answer_match:
        answer = answer_match.group(1)
    else:
        answer = None
    return think, answer

def process_record(record: Dict[str, Any], client: OpenAI, data_folder: str, data_source: str) -> Dict[str, Any]:
    messages = record["messages"]
    idx = record["id"]
    openai_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    ground_truth = ""
    question = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        new_content = []
        for item in content:
            if item["type"] == "text":
                if role == "user":
                    question = item["text"]
                    question = handle_question(question)
                    new_content.append({"type": "text", "text": question})
                elif role == "assistant":
                    ground_truth = item["text"]
            elif item["type"] == "image_url":
                image_path = item["image_url"]['url']
                image_path = os.path.join(data_folder, image_path)
                new_content.append({"type": "image_url", "image_url": {"url": encode_local_image_to_data_url(image_path)}})
        if role == "user":
            openai_messages.append({"role": role, "content": new_content})
    
    payload = {}
    payload["messages"] = openai_messages
    payload["model"] = MODEL_VERSION
    payload["max_tokens"] = 16384
    payload["temperature"] = 1.0
    # print_oai_messages(openai_messages)
    success = False
    for _ in range(3):
        try:
            response = client.chat.completions.create(**payload)
            output_text = response.choices[0].message.content
            if output_text is None:
                continue
            success = True
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
    
    try:
        if not success:
            output_text = ""
        think, answer = handle_output_text(output_text)
        # print(f"Question: {question} \n Answer: {ground_truth}")
        original_answer = ground_truth
        ground_truth = extract_anwser_tag(ground_truth)
        if think is None:
            score = 0.0
        else:
            output_text = f"<think>{think}</think><answer>{answer}</answer>"
            score_dict = compute_score(data_source=data_source, solution_str=output_text, extra_info={"question": question}, ground_truth=ground_truth)
            score = score_dict["acc_score"]
    except Exception as e:
        print(f"Error: {e} when handling output text")
        import traceback; traceback.print_exc()
        output_text = ""
        score = 0.0

    data_dict = {
        "response": output_text,
        "ground_truth": ground_truth,
        # "original_answer": original_answer,
        "score": score,
        # "question": question,
        "idx": idx,
        # "answer": answer,
        # "think": think,
    }

    return data_dict


def process_dataset(
    ds: Dataset,
    data_folder: str,
    data_source: str,
    cache_path: str,
    n: int = 1,
):
    client = get_client()
    if os.path.exists(cache_path):
        cached_results = []
        with open(cache_path, "r") as f:
            for line in f:
                cached_results.append(json.loads(line))
    else:
        cached_results = []
    cached_idx = set([result["idx"] for result in cached_results])
    with ThreadPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
        futures = []
        for record in ds:
            if n > 1:
                record["id"] = f"{record['id']}_roll{record['roll_id']}"
            if record["id"] in cached_idx:
                continue
            futures.append(executor.submit(process_record, record, client, data_folder, data_source))
        pbar = tqdm(total=len(futures))
        for future in as_completed(futures):
            result = future.result()
            pbar.update(1)
            if not result["response"] == "":
                cached_results.append(result)
                with open(cache_path, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return cached_results



def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill responses using an OpenAI-compatible client over a parquet dataset with single-round messages."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config specifying datasets list.")
    parser.add_argument("--num-processes", type=int, default=1, help="Number of processes to use.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples to process.")
    parser.add_argument("--output-folder", type=str, help="Path to save the results.")
    parser.add_argument("--num_rollouts", "-n", type=int, default=1, help="Number of times to roll for each question (default: 1).")

    return parser.parse_args()


CLEAN_SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)

def main(args) -> int:

    config = load_yaml_config(args.config)
    datasets_cfg = config.get("datasets", [])
    if not datasets_cfg:
        print("No datasets specified in config.", file=sys.stderr)
        return 1

    for ds_item in datasets_cfg:
        path = ds_item.get("path")
        data_folder = ds_item.get("data_folder")

        print(f"Processing dataset {path}")

        ds = load_parquet_dataset(path)
        if args.max_samples:
            ds = ds.select(range(args.max_samples))
        
        # Add a roll id to represent this sample is the i-th rollout
        roll_id = []
        for i in range(args.num_rollouts):
            roll_id.extend([i] * len(ds))
        ds = ds.repeat(args.num_rollouts)
        ds = ds.add_column("roll_id", roll_id)
        data_source = os.path.basename(path).split(".")[0]
        save_path = os.path.join(args.output_folder, data_source)
        cache_path = os.path.join(args.output_folder, data_source, "cache.jsonl")
        os.makedirs(save_path, exist_ok=True)
        results = process_dataset(ds, data_folder, data_source=data_source, cache_path=cache_path, n=args.num_rollouts)
        if args.num_rollouts > 1:
            idx2result = collections.defaultdict(list)
            for result in results:
                correct_index = result["idx"].split("_roll")[0]
                if len(idx2result[correct_index]) >= args.num_rollouts:
                    continue
                idx2result[correct_index].append(result)
        else:
            idx2result = {result["idx"]: [result] for result in results}
        original_ds = load_parquet_dataset(path)

        def gen():
            for row in original_ds:
                results = idx2result[row["id"]]
                for result in results:
                    new_messages = []
                    if result["score"] == 0:
                        continue
                    messages = row["messages"]
                    idx = row["id"]
                    system_content = [{"role": "system", "content": [{"type": "text", "text": CLEAN_SYSTEM_PROMPT}]}]
                    new_messages.extend(system_content)
                    for message in messages:
                        role = message["role"]
                        for cont in message["content"]:
                            if "precomputed_tokens" in cont:
                                cont.pop("precomputed_tokens")
                        if role == "user":
                            new_messages.append(message)
                
                    assistant_content = [{"role": "assistant", "content": [{"type": "text", "text": result["response"]}]}]
                    new_messages.extend(assistant_content)
                    yield {"id": idx, "messages": new_messages}

        distilled_dataset = Dataset.from_generator(gen)
        distilled_dataset.save_to_disk(save_path)
    
    return 0


if __name__ == "__main__":
    args = parse_arguments()
    main(args)


