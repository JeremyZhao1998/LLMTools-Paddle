import os
import re
import json
import argparse
import random
import holidays
from tqdm import tqdm
from datasets import load_dataset
from itertools import permutations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./data')
    parser.add_argument("--dataset_name", type=str, default='en')
    parser.add_argument("--subset_name", type=str, default='calendar')
    parser.add_argument("--seed", type=int, default=75)
    parser.add_argument("--shard_size", type=int, default=100000)
    parser.add_argument("--num_shards", type=int, default=10)
    parsed_args = parser.parse_args()
    return parsed_args


def check_three_nums(nums):
    assert len(nums) == 3
    perms = list(permutations(nums))
    for (a, b, c) in perms:
        if a + b == c or a - b == c or a * b == c or (b != 0 and a / b == c):
            return True
    return False


def calculator_check(text):
    text_tokens = text.split()
    # Extract numbers from the text: numbers: {idx_in_text: number_value}
    numbers = {}
    for i, token in enumerate(text_tokens):
        match = re.search(r"^[-+]?\d+(\.\d+)?%?$", token)
        if match:
            num_str = match.group()
            numbers[i] = float(num_str) if "%" not in token else float(num_str[:-1]) / 100
    # Check whether the text contains at least three numbers within a window of 100 tokens, where
    # one of these numbers is the result of applying a mathematical operation to the other two.
    # The mathematical operations are addition, subtraction, multiplication, and division.
    if len(numbers) >= 3:
        num_idx = list(numbers.keys())
        for i in range(len(num_idx) - 2):
            if num_idx[i + 2] - num_idx[i] < 100:
                nums = [numbers[num_idx[i]], numbers[num_idx[i + 1]], numbers[num_idx[i + 2]]]
                if nums[0] < 100 and nums[1] < 100 and nums[2] < 100:
                    continue
                if check_three_nums(nums):
                    # print("Three numbers found: ", nums)
                    return True
    return False


def check_random_drop(text):
    if len(text.split()) > 5:
        return random.random() < 0.05
    return False


def translator_check(text):
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    german_pattern = re.compile(r'[äöüßÄÖÜ]')
    spanish_pattern = re.compile(r'[ñáéíóúüÁÉÍÓÚÜ]')
    vietnamese_pattern = re.compile(r'[ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯư]')
    hindi_pattern = re.compile(r'[\u0900-\u097F]')
    chinese_pattern = re.compile(r'[\u4E00-\u9FFF]')
    if arabic_pattern.search(text) or german_pattern.search(text) or spanish_pattern.search(text) or \
            vietnamese_pattern.search(text) or hindi_pattern.search(text) or chinese_pattern.search(text):
        return True
    return False


def calendar_check(text):
    date_patterns = [
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',  # 2024-07-08 or 2024/07/08
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # 08-07-2024 or 08/07/24
        r'\b\d{1,2} ?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* ?\d{2,4}\b',  # 8 Jul 2024 or 8 July 2024
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* \d{1,2},? \d{4}\b',  # July 8, 2024 or July 8 2024
        r'(\d+)\s+days\s+ago',  # x days ago
        r'in\s+(\d+)\s+days',  # in x days
    ]
    special_dates = [
        'tomorrow',
        'today',
        'yesterday',
        'the day after tomorrow',
        'the day before yesterday',
    ]
    all_holidays = holidays.country_holidays('US', years=2024)
    for date, name in all_holidays.items():
        special_dates.append(name)
    for pattern in date_patterns:
        if re.search(pattern, text):
            return True
    for date in special_dates:
        if date.lower() in text.lower():
            return True
    return False


def process_split(dataset, output_dir, check_fn, split="train"):
    assert split in ["train", "validation"]
    file_cnt, text_cnt = 0, 0
    output_file_name = os.path.join(str(output_dir), "c4-" + split + f"-{file_cnt}.json")
    output_file = open(output_file_name, "w")
    for data in tqdm(dataset[split], total=LEN_C4_TRAIN if split == "train" else LEN_C4_VALIDATION,
                     desc="Extracting subset " + args.subset_name + " from C4 " + split + " split"):
        data_split = data["text"].split("\n")
        if file_cnt == args.num_shards:
            break
        for text in data_split:
            if check_fn(text):
                json.dump({"text": text}, output_file)
                output_file.write("\n")
                text_cnt += 1
                if text_cnt % args.shard_size == 0:
                    output_file.close()
                    print(f"Text writing to file {output_file_name} completed. Total texts: {text_cnt}")
                    file_cnt += 1
                    if file_cnt == args.num_shards:
                        break
                    output_file_name = os.path.join(str(output_dir), "c4-" + split + f"-{file_cnt}.json")
                    output_file = open(output_file_name, "w")
    print(f"Extracted {text_cnt} texts from C4 {split} split")


def main():
    dataset = load_dataset(path=os.path.join(args.data_root, "c4"), name="en", streaming=True)
    output_dir = os.path.join(args.data_root, "c4", args.subset_name)
    os.makedirs(output_dir, exist_ok=True)
    if args.subset_name == 'calculator':
        check_fn = calculator_check
    elif args.subset_name == 'qa' or args.subset_name == 'search':
        check_fn = check_random_drop
    elif args.subset_name == 'translator':
        check_fn = translator_check
    elif args.subset_name == 'calendar':
        check_fn = calendar_check
    else:
        raise ValueError(f"Invalid subset name: {args.subset_name}")
    # process_split(dataset, output_dir, check_fn, "validation")
    process_split(dataset, output_dir, check_fn, "train")


if __name__ == '__main__':
    LEN_C4_TRAIN = 364868892
    LEN_C4_VALIDATION = 364608
    args = parse_args()
    main()
