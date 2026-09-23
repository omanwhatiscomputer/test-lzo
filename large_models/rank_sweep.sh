# Rank-diagnostic sweep: LOZO with --rank_probe over tasks x r x nu x seeds.
# Usage: MODEL=facebook/opt-1.3b STEPS=20000 bash rank_sweep.sh
# Extra args are forwarded to lozo.sh / run_lozo.py.

TASKS=${TASKS:-"SST2 RTE"}
RANKS=${RANKS:-"2 4 8 16"}
INTERVALS=${INTERVALS:-"50 100"}
SEEDS=${SEEDS:-"0 1 2"}
# True-gradient reference for a few OPT layers (set GRAD_LAYERS="" to disable)
GRAD_LAYERS=${GRAD_LAYERS-'layers\.(0|6|12|18|23)\.(self_attn\.(q_proj|v_proj)|fc1)\.weight$'}

GRAD_ARGS=""
if [ -n "$GRAD_LAYERS" ]; then
    GRAD_ARGS="--rank_probe_grad_layers $GRAD_LAYERS"
fi

for TASK in $TASKS; do
for RANK in $RANKS; do
for STEP_INTERVAL in $INTERVALS; do
for SEED in $SEEDS; do
    MODEL_NAME=${MODEL:-facebook/opt-1.3b}; MODEL_NAME=${MODEL_NAME##*/}
    # lozo.sh's SEED only picks the training subset; --seed also varies the LOZO perturbation stream.
    # Separate output_dir so these runs never resume from a non-probe checkpoint.
    TASK=$TASK RANK=$RANK STEP_INTERVAL=$STEP_INTERVAL SEED=$SEED \
        bash lozo.sh --rank_probe $GRAD_ARGS --seed $SEED \
        --output_dir result/rankprobe/$TASK-$MODEL_NAME-r$RANK-nu$STEP_INTERVAL-s$SEED "$@"
done
done
done
done
