DATA_ROOT=<path/to/your/data/root>
DATASET_NAME='c4/qa'
CONVERTED_NAME=${DATASET_NAME}'_star'

python main.py \
--mode 'convert' \
--model_path <path/to/your/model> \
--data_root ${DATA_ROOT} \
--dataset_name ${DATASET_NAME} \
--tao_s 0.5 \
--tao_f 0.1 \
--max_input_len 128 \
--max_output_len 128 \
--start_file_cnt 0 \
--start_data_cnt 0 \
--converted_size 10000 \
--shard_size 1000 \
--batch_size 8 \
--num_workers 4 \
--print_freq 100 \

echo "Making directory: $DATA_ROOT/$CONVERTED_NAME/tokenized"
mkdir -p "$DATA_ROOT/$CONVERTED_NAME/tokenized"

for file in $DATA_ROOT/$CONVERTED_NAME/*.jsonl; do
    python -u  create_pretraining_data.py \
        --model_name "meta-llama/Llama-2-7b-chat/" \
        --tokenizer_name "LlamaTokenizer" \
        --data_format "JSON" \
        --input_path "$file" \
        --append_eos \
        --output_prefix "$DATA_ROOT/$CONVERTED_NAME/tokenized/$(basename "$file")"  \
        --workers 1 \
        --log_interval 5 \
        --data_impl "mmap"
done

python merge_pretraining_data.py \
    --input "$DATA_ROOT/$CONVERTED_NAME/tokenized" \
    --output-prefix "$DATA_ROOT/$CONVERTED_NAME/merged" \
    --data_impl mmap
