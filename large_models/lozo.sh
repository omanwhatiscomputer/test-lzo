MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-16}
EPS=${EPS:-1e-3}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-4000}

MODE=${MODE:-ft}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora"
fi

LR=${LR:-1e-7}
TASK=${TASK:-DROP}
SEED=${SEED:-0}
RANK=${RANK:-1}
RANK_SCHEDULE=${RANK_SCHEDULE:-} # e.g. "4:10000,2" = rank 4 for 10000 steps, then 2; overrides RANK
STEP_INTERVAL=${STEP_INTERVAL:-100}
Tainer=LOZO

RANK_TAG=$RANK
if [ -n "$RANK_SCHEDULE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --rank_schedule $RANK_SCHEDULE"
    RANK_TAG=$(echo "$RANK_SCHEDULE" | tr ':,' 'x-') # e.g. 4x10000-2
fi

case $TASK in
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
    *)
        TASK_ARGS=""
        ;;
esac

TAG=$Tainer-$MODE-$STEPS-$BS-$LR-$EPS-$SEED-$STEP_INTERVAL-$RANK_TAG


echo $TAG
echo "Task: $TASK"
echo "BS: $BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
echo "RANK: ${RANK_SCHEDULE:-$RANK}"
echo "STEP INTERVAL: $STEP_INTERVAL"

python run_lozo.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer $Tainer --load_float16 \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
    --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
    --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
    --train_as_classification \
    --step_interval $STEP_INTERVAL \
    --rank_r $RANK \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@" 





