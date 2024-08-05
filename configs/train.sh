# python -u  -m paddle.distributed.launch --gpus "0,1" pretrain.py <your/config/path>
python -u  -m paddle.distributed.launch --gpus "0,1" pretrain.py ./configs/train_calendar.json
