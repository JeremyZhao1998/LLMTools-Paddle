python main.py \
--mode 'eval' \
--model_path <path/to/your/model> \
--data_root <path/to/your/data/root> \
--output_path './outputs/eval_translator/llama_mlqa_zh' \
--dataset_name 'mlqa-zh' \
--max_input_len 512 \
--max_output_len 64 \
--batch_size 8 \
--print_freq 10 \
--num_workers 4 \
