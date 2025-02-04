python main.py \
--mode 'eval' \
--model_path <path/to/your/model> \
--data_root <path/to/your/data/root> \
--output_path './outputs/eval_calculator/llama_asdiv' \
--dataset_name 'asdiv' \
--max_input_len 128 \
--max_output_len 32 \
--batch_size 16 \
--print_freq 10 \
--num_workers 4 \
