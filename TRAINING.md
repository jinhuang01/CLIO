CLIO's training:
``` 
python3 imagenet.py
    --data=<Data-Path> \
    --train-batch=<train-batch-size> \
    --test-batch=<test-batch-size> \
    --epochs=<epochs> \
    --lr=<learning-rate> \
    --workers=<worker-number> \
    --gpu-id=<gpu-ids> \
    --widths=<widths-for-clio> \
    --default-width=<default-width-for-clio> \
    --use-random-connect \
    --split-point=<default-partition-point> \
    --resume-path=<base-model-checkpoint>
```