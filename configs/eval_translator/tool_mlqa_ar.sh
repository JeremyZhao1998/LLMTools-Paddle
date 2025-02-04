python main.py \
--mode 'eval' \
--model_path <path/to/your/trained_model> \
--data_root <path/to/your/data/root> \
--output_path './outputs/eval_translator/tool_mlqa_ar' \
--dataset_name 'mlqa-ar' \
--max_input_len 512 \
--max_output_len 64 \
--batch_size 8 \
--print_freq 10 \
--num_workers 4 \
