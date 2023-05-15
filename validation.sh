MODEL=caformer_s18
python3 ./validate.py /path/to/imagenet  --model $MODEL -b 128 \
  --checkpoint /path/to/checkpoint 
