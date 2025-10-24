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
    --checkpointing-steps=2500 \
    --max-train-steps=50001 \
    --subject="[1]" \
    --valid-sub=1 \
    --unseen-sub=9 \
    --alpha=0 \
    --beta=0 \
    --hour=36 \
    --encoder="vae" \
    --hidden-dim=4096 \
    --vae-path="vae-nsd-s1-vs1-bs24-350" \
    --model-depth=8 \
    --model-head=13 \
    --mode \
    --exp-name="fm-vae-s1-vs1-d8-h13-bs24-es350-x-mode" \
    # --resume \
    # --resume-id="jqt0dv7w"

    # nohup bash run_sit_os.sh > logs/fm_s1_down2_x.log 2>&1 &