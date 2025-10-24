python train_b2i_os.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --prediction="x" \
    --output-dir="/home/bingxing2/ailab/group/ai4neuro/BrainVL/BrainSyn/train_logs" \
    --wandb-log \
    --batch-size=24 \
    --sampling-steps=10000 \
    --checkpointing-steps=1000 \
    --max-train-steps=40001 \
    --alpha=0 \
    --beta=0 \
    --encoder="vae" \
    --unseen-sub=9 \
    --subject="[2]" \
    --valid-sub=2 \
    --hour=1 \
    --vae-path="ft-vae-ps1-s2-1h-all-bs24" \
    --model-depth=8 \
    --model-head=13 \
    --finetune \
    --ft_mode="mlp" \
    --fm-path="fm-vae-s1-vs1-d8-h13-bs24-es350-x-mode" \
    --mode \
    --exp-name="ft-fm-ps1-s2-1h-mlp-bs24-x-mode" \
    # --resume \
    # --resume-id="v6kdfz2m"

    # nohup bash run_sit_os_bwd_ft.sh > logs/ft_fm_ps1_s7_1h_mlp.log 2>&1 &