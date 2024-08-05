import re
import os
import time
import json
import argparse

import paddle
from paddlenlp.transformers import LlamaTokenizer

from eval_benchmarks.benchmarks import MathBenchmarks, QABenchmarks, SearchQA, MultiLanguageQA, DateBenchmark
from train_dataset.dataset import FinetuneDataset
from llm_models.llama2 import Llama2
from llm_models.prompts import PROMPTS
from tool_api_calls import calculator, qa, search, translator, calendar


def parse_args():
    parser = argparse.ArgumentParser()
    # Mode configuration
    parser.add_argument("--mode", type=str, default='convert', choices=['convert', 'eval'])
    # Model configuration
    parser.add_argument("--model_path", type=str, default='meta-llama/Llama-2-7b-chat')
    parser.add_argument('--max_input_len', type=int, default=128)
    parser.add_argument("--max_output_len", type=int, default=32)
    # Data configuration
    parser.add_argument('--data_root', type=str, default='/network_space/server129/zhaozijing/datasets')
    parser.add_argument('--dataset_name', type=str, default='c4/calendar')
    parser.add_argument('--tao_s', type=float, default=0.5)
    parser.add_argument('--tao_f', type=float, default=0.1)
    parser.add_argument('--K', type=float, default=5)
    parser.add_argument('--converted_size', type=int, default=10000)
    parser.add_argument('--shard_size', type=int, default=1000)
    parser.add_argument('--start_file_cnt', type=int, default=0)
    parser.add_argument('--start_data_cnt', type=int, default=100)
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_path", type=str, default='./outputs')
    # Other configuration
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--flush", type=int, default=1)
    parsed_args = parser.parse_args()
    parsed_args.flush = bool(parsed_args.flush)
    return parsed_args


def build_tokenizer():
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    return tokenizer


def build_benchmark():
    if args.dataset_name in ['asdiv', 'svamp']:
        return MathBenchmarks(args.data_root, args.dataset_name)
    elif args.dataset_name in ['squad', 'google_re']:
        return QABenchmarks(args.data_root, args.dataset_name)
    elif args.dataset_name in ['web_qs', 'nq']:
        return SearchQA(args.data_root, args.dataset_name)
    elif args.dataset_name.startswith('mlqa'):
        return MultiLanguageQA(args.data_root, args.dataset_name)
    elif args.dataset_name == 'dateset':
        return DateBenchmark(args.data_root, args.dataset_name)


def build_model():
    start = time.time()
    model = Llama2.from_pretrained(args.model_path)
    if args.max_output_len is not None:
        model.config.max_new_tokens = args.max_output_len
    end = time.time()
    print(f"Load model finished. Time cost: {end - start} seconds", flush=args.flush)
    return model


def get_api_func():
    if args.dataset_name in ['svamp', 'mawps', 'asdiv', 'c4/calculator']:
        return calculator
    if args.dataset_name in ['squad', 'google_re', 'trex', 'concept_net', 'c4/qa']:
        return qa
    if args.dataset_name in ['web_qs', 'nq', 'trivia_qa', 'c4/search']:
        return search
    if args.dataset_name.startswith('mlqa') or args.dataset_name == 'c4/translator':
        return translator
    if args.dataset_name == 'dateset' or args.dataset_name == 'c4/calendar':
        return calendar


def clean_outputs(outputs):
    # Replace the '<unk>' tokens at the end of the outputs
    outputs = [re.sub(r'(<unk>)+$', '', output) for output in outputs]
    # Replace all '\n' tokens with ' ' in the outputs
    outputs = [output.replace('\n', ' ') for output in outputs]
    # Replace all '</s>' tokens with ' ' in the outputs
    outputs = [output.replace('</s>', ' ') for output in outputs]
    return outputs


def batch_generate(model, tokenizer, input_list, eos_token_id=None):
    list_len = len(input_list)
    batch_idx_list = list(range(0, list_len, args.batch_size))
    output_list = []
    for b in batch_idx_list:
        bs = min(args.batch_size, list_len - b)
        inputs = tokenizer(input_list[b: b + bs], return_tensors='pd', padding=True)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            attention_mask=inputs["attention_mask"],
            eos_token_id=eos_token_id if eos_token_id is not None else tokenizer.eos_token_id,
        )
        outputs = tokenizer.batch_decode(outputs[0])
        output_list.extend(outputs)
    output_list = clean_outputs(output_list)
    return output_list


def weighted_loss(model, tokenizer, seq, start_idx, prefix=''):
    seq_token_ids = tokenizer(seq)["input_ids"][1:]
    weights = paddle.to_tensor([max(0.0, 1.0 - 0.2 * t) for t in range(len(seq_token_ids) - start_idx)])
    weights = weights / paddle.sum(weights)
    gt_token_ids = seq_token_ids[start_idx:]
    tokenized_inputs = tokenizer(prefix + seq, return_tensors='pd')
    try:
        logits = model(
            input_ids=tokenized_inputs["input_ids"],
            position_ids=tokenized_inputs["position_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
        )[0][0]
        logits = logits.to(paddle.float32)
    except OSError:
        print("Error in processing: " + prefix + seq)
        return paddle.to_tensor(1000.0)
    probs = paddle.nn.functional.softmax(logits)[-weights.shape[0] - 1: -1, :]
    probs = probs[:, gt_token_ids].diagonal()
    loss = -paddle.sum(paddle.log(probs) * weights)
    return loss


@paddle.no_grad()
def convert_dataset():
    tokenizer = build_tokenizer()
    api_call_id = tokenizer("[")["input_ids"][1]
    api_finish_id = tokenizer("]")["input_ids"][1]
    prompt = PROMPTS[args.dataset_name]
    model = build_model()
    model.eval()
    file_cnt, output_cnt = args.start_file_cnt, args.start_file_cnt * args.shard_size
    output_file_name = os.path.join(str(args.data_root), args.dataset_name + "_star", f"star-{file_cnt}.jsonl")
    if not os.path.exists(os.path.dirname(output_file_name)):
        os.makedirs(os.path.dirname(output_file_name))
    output_file = open(output_file_name, "w")
    dataset = FinetuneDataset(args.data_root, args.dataset_name)
    cur_time = time.time()
    for i in range(args.start_data_cnt, len(dataset)):
        if i % args.print_freq == 0:
            print('------------------------------------------', flush=args.flush)
            print(f"Sample: {i} / {len(dataset)}", flush=args.flush)
            print(f"Selected: {output_cnt}", flush=args.flush)
            print('Time cost: ', time.time() - cur_time, 's', flush=args.flush)
            cur_time = time.time()
            print('------------------------------------------', flush=args.flush)
        sample = dataset[i]
        sample_token_ids = tokenizer(sample)["input_ids"][1:]
        if 0 < len(sample_token_ids) <= args.max_input_len:
            input_prefix = prompt + "Input: " + sample + "\nOutput: "
            inputs = tokenizer(input_prefix + sample, return_tensors='pd')
            logits = model(
                input_ids=inputs['input_ids'],
                position_ids=inputs['position_ids'],
                attention_mask=inputs['attention_mask'],
            )[0][0]
            probs = paddle.nn.functional.softmax(logits)
            api_call_probs = probs[-(len(sample_token_ids) + 1):, api_call_id]
            topk_probs, topk_idx = paddle.topk(api_call_probs, k=args.K)
            topk_idx = [i.item() for i in topk_idx[topk_probs > args.tao_s] if i < len(sample_token_ids)]
            if len(topk_idx) > 0:
                sample_subseq = tokenizer.batch_decode([sample_token_ids[: i] for i in topk_idx])
                sample_res = tokenizer.batch_decode([sample_token_ids[i:] for i in topk_idx])
                selected_list = [input_prefix + sub_s + "[" for sub_s in sample_subseq]
                output_list = batch_generate(model, tokenizer, selected_list, eos_token_id=api_finish_id)
                output_list = ["[ " + (s[: s.find(']') + 1] if ']' in s else s + ']') for s in output_list]
                api_result_list = [api_func(s) for s in output_list]
                api_result_list = clean_outputs(api_result_list)
                selected_samples, reduce_values = [], []
                for j, (idx, call, result) in enumerate(zip(topk_idx, output_list, api_result_list)):
                    loss_ori = weighted_loss(model, tokenizer, sample, idx, prefix='')
                    loss_call = weighted_loss(model, tokenizer, sample, idx, prefix=call)
                    if isinstance(result + sample, str) and 0 < len(result + sample) < 10000:
                        loss_result = weighted_loss(model, tokenizer, sample, idx, prefix=result)
                        reduce_value = min(loss_ori, loss_call) - loss_result
                        if reduce_value > args.tao_f:
                            selected_samples.append(sample_subseq[j] + call + sample_res[j])
                            reduce_values.append(reduce_value)
                if len(selected_samples) > 0:
                    idx = paddle.argmax(paddle.to_tensor(reduce_values))
                    selected_sample = selected_samples[idx]
                    json.dump({"text": selected_sample}, output_file)
                    output_file.write("\n")
                    output_cnt += 1
                    if output_cnt % args.shard_size == 0:
                        output_file.close()
                        print('------------------------------------------', flush=args.flush)
                        print(f"Texts writing to file {output_file_name} completed. Total texts: {output_cnt}")
                        print('------------------------------------------', flush=args.flush)
                        if output_cnt >= args.converted_size:
                            break
                        file_cnt += 1
                        output_file_name = os.path.join(str(args.data_root), args.dataset_name + "_star",
                                                        f"star-{file_cnt}.jsonl")
                        output_file = open(output_file_name, "w")


"""
@paddle.no_grad()
def convert_stage1():
    tokenizer = build_tokenizer()
    api_call_id = tokenizer("[")["input_ids"][1]
    api_finish_id = tokenizer("]")["input_ids"][1]
    prompt = PROMPTS[args.dataset_name]
    model = build_model()
    model.eval()
    file_cnt, output_cnt = args.start_file_cnt, args.start_file_cnt * args.shard_size
    output_file_name = os.path.join(str(args.data_root), args.dataset_name + "_star", f"tmp-stage1-{file_cnt}.json")
    if not os.path.exists(os.path.dirname(output_file_name)):
        os.makedirs(os.path.dirname(output_file_name))
    output_file = open(output_file_name, "w")
    dataset = FinetuneDataset(args.data_root, args.dataset_name)
    cur_time = time.time()
    for i in range(args.start_data_cnt, len(dataset)):
        if i % args.print_freq == 0:
            print('------------------------------------------', flush=args.flush)
            print(f"Sample: {i} / {len(dataset)}", flush=args.flush)
            print('Time cost: ', time.time() - cur_time, 's', flush=args.flush)
            cur_time = time.time()
        sample = dataset[i]
        sample_token_ids = tokenizer(sample)["input_ids"][1:]
        if len(sample_token_ids) <= args.max_input_len:
            input_prefix = prompt + "Input: " + sample + "\nOutput: "
            inputs = tokenizer(input_prefix + sample, return_tensors='pd')
            logits = model(
                input_ids=inputs['input_ids'],
                position_ids=inputs['position_ids'],
                attention_mask=inputs['attention_mask'],
            )[0][0]
            probs = paddle.nn.functional.softmax(logits)
            api_call_probs = probs[-(len(sample_token_ids) + 1):, api_call_id]
            topk_probs, topk_idx = paddle.topk(api_call_probs, k=args.K)
            topk_idx = [i.item() for i in topk_idx[topk_probs > args.tao_s] if i < len(sample_token_ids)]
            if len(topk_idx) > 0:
                sample_subseq = tokenizer.batch_decode([sample_token_ids[: i] for i in topk_idx])
                selected_list = [input_prefix + sub_s + "[" for sub_s in sample_subseq]
                output_list = batch_generate(model, tokenizer, selected_list, eos_token_id=api_finish_id)
                output_list = ["[ " + s[: s.find(']')] for s in output_list]
                json.dump({"sample_idx": i, "topk_idx": topk_idx, "output_list": output_list}, output_file)
                output_file.write("\n")
                output_cnt += 1
                if output_cnt % args.shard_size == 0:
                    output_file.close()
                    print('------------------------------------------', flush=args.flush)
                    print(f"Tmp texts writing to file {output_file_name} completed. Total texts: {output_cnt}")
                    file_cnt += 1
                    output_file_name = os.path.join(str(args.data_root), args.dataset_name + "_star",
                                                    f"tmp-stage1-{file_cnt}.json")
                    output_file = open(output_file_name, "w")
    output_file.close()


def convert_stage2():
    cur_time = time.time()
    stage1_file_names = [name for name in os.listdir(os.path.join(str(args.data_root), args.dataset_name + "_star"))
                         if name.startswith("tmp-stage1")]
    for stage1_file_name in stage1_file_names:
        stage1_file = open(os.path.join(str(args.data_root), args.dataset_name + "_star", stage1_file_name), "r")
        stage2_file_name = stage1_file_name.replace("stage1", "stage2")
        stage2_file = open(os.path.join(str(args.data_root), args.dataset_name + "_star", stage2_file_name), "w")
        for line in stage1_file:
            data = json.loads(line)
            output_list = data["output_list"]
            api_result_list = [api_func(s) for s in output_list]
            json.dump({
                "sample_idx": data['sample_idx'],
                "topk_idx": data["topk_idx"],
                "output_list": output_list,
                "api_result_list": api_result_list
            }, stage2_file)
            stage2_file.write("\n")
        stage1_file.close()
        stage2_file.close()
        print('------------------------------------------', flush=args.flush)
        print(f"Tmp texts writing to file {stage2_file_name} completed.")
        print('Time cost: ', time.time() - cur_time, 's', flush=args.flush)
        cur_time = time.time()


@paddle.no_grad()
def convert_stage3():
    tokenizer = build_tokenizer()
    model = build_model()
    model.eval()
    file_cnt, output_cnt = args.start_file_cnt, args.start_file_cnt * args.shard_size
    output_file_name = os.path.join(str(args.data_root), args.dataset_name + "_star", f"star-{file_cnt}.jsonl")
    output_file = open(output_file_name, "w")
    dataset = FinetuneDataset(args.data_root, args.dataset_name)
    cur_time = time.time()
    stage2_file_names = [name for name in os.listdir(os.path.join(str(args.data_root), args.dataset_name + "_star"))
                         if name.startswith("tmp-stage2")]
    for stage2_file_name in stage2_file_names:
        stage2_file = open(os.path.join(str(args.data_root), args.dataset_name + "_star", stage2_file_name), "r")
        for line in stage2_file:
            data = json.loads(line)
            i = data["sample_idx"]
            topk_idx = data["topk_idx"]
            output_list = data["output_list"]
            api_result_list = data["api_result_list"]
            if i % args.print_freq == 0:
                print('------------------------------------------', flush=args.flush)
                print(f"Sample: {i} / {len(dataset)}", flush=args.flush)
                print('Time cost: ', time.time() - cur_time, 's', flush=args.flush)
                cur_time = time.time()
            sample = dataset[i]
            sample_token_ids = tokenizer(sample)["input_ids"][1:]
            sample_subseq = tokenizer.batch_decode([sample_token_ids[: i] for i in topk_idx])
            sample_res = tokenizer.batch_decode([sample_token_ids[i:] for i in topk_idx])
            selected_samples, reduce_values = [], []
            for j, (idx, call, result) in enumerate(zip(topk_idx, output_list, api_result_list)):
                loss_ori = weighted_loss(model, tokenizer, sample, idx, prefix='')
                loss_call = weighted_loss(model, tokenizer, sample, idx, prefix=call)
                loss_result = weighted_loss(model, tokenizer, sample, idx, prefix=result)
                reduce_value = min(loss_ori, loss_call) - loss_result
                if reduce_value > args.tao_f:
                    selected_samples.append(sample_subseq[j] + call + sample_res[j])
                    reduce_values.append(reduce_value)
            if len(selected_samples) > 0:
                idx = paddle.argmax(paddle.to_tensor(reduce_values))
                selected_sample = selected_samples[idx]
                json.dump({"text": selected_sample}, output_file)
                output_file.write("\n")
                output_cnt += 1
                if output_cnt % args.shard_size == 0:
                    output_file.close()
                    print('------------------------------------------', flush=args.flush)
                    print(f"Texts writing to file {output_file_name} completed. Total texts: {output_cnt}")
                    file_cnt += 1
                    output_file_name = os.path.join(str(args.data_root), args.dataset_name + "_star",
                                                    f"star-{file_cnt}.jsonl")
                    output_file = open(output_file_name, "w")
                if output_cnt == args.converted_size:
                    break
    output_file.close()"""


@paddle.no_grad()
def evaluate():
    tokenizer = build_tokenizer()
    api_finish_id = tokenizer("]")["input_ids"][1]
    benchmark_dataset = build_benchmark()
    model = build_model()
    correct = 0
    output_list = []
    cur_time = time.time()
    batch_idx_list = list(range(0, len(benchmark_dataset), args.batch_size))
    for batch_idx in batch_idx_list:
        bs = min(args.batch_size, len(benchmark_dataset) - batch_idx)
        input_list = [sample[0] for sample in benchmark_dataset[batch_idx: batch_idx + bs]]
        outputs = batch_generate(model, tokenizer, input_list, eos_token_id=api_finish_id)
        outputs = [api_func(output, complete=False) if api_func == translator else api_func(output)
                   for output in outputs]
        new_input_list = [input_s + ' ' + output_s for input_s, output_s in zip(input_list, outputs)]
        new_outputs = batch_generate(model, tokenizer, new_input_list)
        final_outputs = [o + n_o for o, n_o in zip(outputs, new_outputs)]
        output_list.extend(final_outputs)
        if (batch_idx // args.batch_size) % args.print_freq == 0:
            print('------------------------------------------', flush=args.flush)
            print(f"Sample: {batch_idx} / {len(benchmark_dataset)}", flush=args.flush)
            print(f"Question: {re.sub(r'(<unk>)+$', '', input_list[-1])}", flush=args.flush)
            print(f"Model Output: {re.sub(r'(<unk>)+$', '', final_outputs[-1])}", flush=args.flush)
            print(f'GT: {benchmark_dataset.data[batch_idx: batch_idx + bs][-1][1]}', flush=args.flush)
            print('Time cost: ', time.time() - cur_time, 's', flush=args.flush)
            print('------------------------------------------', flush=args.flush)
            cur_time = time.time()
    print('Saving outputs...', flush=args.flush)
    output_file = os.path.join(args.output_path, f'{args.dataset_name}_outputs.txt')
    with open(output_file, 'w') as f:
        for output in output_list:
            f.write(output + '\n')
    print('Outputs saved to ', output_file, flush=args.flush)
    print('Time cost: ', time.time() - cur_time, 's', flush=args.flush)
    cur_time = time.time()
    print('Check answers...', flush=args.flush)
    for i, (output, label) in enumerate(zip(output_list, benchmark_dataset.data)):
        if benchmark_dataset.check_answer(output, label[1]):
            correct += 1
    print(f"Accuracy: {correct / len(output_list)}", flush=args.flush)
    print('Time cost: ', time.time() - cur_time, 's', flush=args.flush)


def main():
    if args.mode == 'eval':
        evaluate()
    elif args.mode == 'convert':
        convert_dataset()


if __name__ == "__main__":
    args = parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}", flush=args.flush)
    api_func = get_api_func()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    main()
