# Smaller Vision Transformers for Performance on Datasets Small in Resolution and Size

Trained models from the results of Swin + DeiT are available in trained_models, and can be loaded and evaulated as, for example:
```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py \
    --cfg trained_models/CIFAR100_long_cosine_scheduler_hard_distill --output trained_models/CIFAR100_long_cosine_scheduler_hard_distill \
    --opts EVAL_MODE True \
```

---
Trained models using Swin Transformer only, `.yaml` file that record hyper-parameters are provided in the `configs` folder. All hyper-parameter setting of the experiment in our report can be find in the folder:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py \
    --cfg configs/swin_32x32_modify.yaml --batch-size 128 --output output/CIFAR100 \
    --accumulation-steps 2 --cache-mode part \
    --opts SAVE_FREQ 10 PRINT_FREQ 1 TRAIN.EPOCHS 200 DATA.DATASET cifar100 &>> output/CIFAR100/Swin_32x32_0_60.log
```

Train from checkpoint from `checkpoint_file_path.pth`:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py \
    --cfg configs/swin_32x32_modify.yaml \
    --resume "{checkpoint_file_path.pth}" \
    --batch-size 128 --output output/CIFAR100 \
    --accumulation-steps 2 --cache-mode part \
    --opts SAVE_FREQ 10 PRINT_FREQ 1 TRAIN.EPOCHS 200 DATA.DATASET cifar100 &>> output/CIFAR100/Swin_32x32_61_200.log
```
