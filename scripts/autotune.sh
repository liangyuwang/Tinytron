#!/bin/bash
# ========================================================
# GPT Auto-Tuner: Optimizing for Maximum tok/sec
# ========================================================
SEP_SIZES_STR=${SEP_SIZES:-"1 2 4 8"}
BATCH_SIZES_STR=${BATCH_SIZES:-"1 2 4 8 16 32"}

TARGET_STEPS=${TARGET_STEPS:-100}
WARMUP_STEPS=${WARMUP_STEPS:-20}
RUN_SCRIPT=${RUN_SCRIPT:-"scripts/debug/pretrain.sh"}

DEBUG=${DEBUG:-0}

SEP_SIZES=($SEP_SIZES_STR)
BATCH_SIZES=($BATCH_SIZES_STR)
TOTAL_REQUIRED_STEPS=$((TARGET_STEPS + WARMUP_STEPS))
echo "======================================================"
echo "⚙️  Auto-Tune Configuration:"
echo "   SEP_SIZES    : ${SEP_SIZES[*]}"
echo "   BATCH_SIZES  : ${BATCH_SIZES[*]}"
echo "   TARGET_STEPS : $TARGET_STEPS"
echo "   WARMUP_STEPS : $WARMUP_STEPS"
echo "   RUN_SCRIPT   : $RUN_SCRIPT"
echo "======================================================"

LOG_FILE="autotune_temp.log"
RESULTS_FILE="autotune_results.csv"

echo "SEP_SIZE,BATCH_SIZE,AVG_TOK_SEC,STATUS" > $RESULTS_FILE
echo -e "\n🔥 Starting Auto-Tune: Target Steps = $TARGET_STEPS (Warmup = $WARMUP_STEPS)...\n"

trap 'echo "🚨 Interrupted! Cleaning up..."; pkill -f "scripts/debug/pretrain.py"; [ -n "$TAIL_PID" ] && kill $TAIL_PID 2>/dev/null; rm -f $LOG_FILE; exit 1' INT

best_tok_sec=0
best_config=""

for sp in "${SEP_SIZES[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        echo "------------------------------------------------------"
        echo "⏳ Testing [ SEP_SIZE=$sp | BATCH_SIZE=$bs ]"
        
        > $LOG_FILE

        export SEP_SIZE=$sp
        export BATCH_SIZE=$bs
        bash $RUN_SCRIPT > $LOG_FILE 2>&1 &
        RUN_PID=$!

        TAIL_PID=""
        if [ "$DEBUG" -eq 1 ]; then
            echo "🐛 [DEBUG MODE] Streaming logs..."
            tail -f $LOG_FILE &
            TAIL_PID=$!
        fi

        STEP_COUNT=0
        STATUS="RUNNING"
        
        while true; do
            if grep -qiE "out of memory|OOM" $LOG_FILE; then
                STATUS="OOM"
                echo "❌ Failed: CUDA Out Of Memory."
                kill -9 $RUN_PID 2>/dev/null
                wait $RUN_PID 2>/dev/null
                break
            fi

            if ! kill -0 $RUN_PID 2>/dev/null; then
                if grep -qiE "error|traceback" $LOG_FILE && ! grep -qiE "out of memory" $LOG_FILE; then
                     STATUS="ERROR"
                     echo "❌ Failed: Script crashed (See $LOG_FILE for details)."
                     if [ "$DEBUG" -eq 0 ]; then
                         echo "Last 10 lines of log:"
                         tail -n 10 $LOG_FILE
                     fi
                else
                    if [ "$STEP_COUNT" -le "$WARMUP_STEPS" ]; then
                         STATUS="INSUFFICIENT_STEPS"
                         echo -e "\n❌ Failed: Script exited early. Only got $STEP_COUNT steps (needed > $WARMUP_STEPS for warmup)."
                     else
                         STATUS="SUCCESS"
                         echo -e "\n⚠️ Warning: Script finished early at step $STEP_COUNT, but gathered enough valid data."
                     fi
                fi
                break
            fi

            STEP_COUNT=$(grep -c "tok/sec:" $LOG_FILE)
            
            if [ "$STEP_COUNT" -ge "$TOTAL_REQUIRED_STEPS" ]; then
                STATUS="SUCCESS"
                echo "✅ Gathered enough steps ($TARGET_STEPS). Terminating run..."
                pkill -P $RUN_PID 2>/dev/null || kill -9 $RUN_PID 2>/dev/null
                wait $RUN_PID 2>/dev/null
                break
            fi
            
            sleep 2
        done

        if [ -n "$TAIL_PID" ]; then
            kill $TAIL_PID 2>/dev/null
        fi

        if [ "$STATUS" == "SUCCESS" ]; then
            avg_tok_sec=$(grep "tok/sec:" $LOG_FILE | \
                awk -F'tok/sec: ' '{print $2}' | awk '{print $1}' | \
                tail -n +$((WARMUP_STEPS + 1)) | \
                awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print 0}')
            
            echo "📊 Average tok/sec (excluding warmup): $avg_tok_sec"
            
            echo "$sp,$bs,$avg_tok_sec,SUCCESS" >> $RESULTS_FILE
            
            if (( $(echo "$avg_tok_sec > $best_tok_sec" | bc -l) )); then
                best_tok_sec=$avg_tok_sec
                best_config="SEP_SIZE=$sp, BATCH_SIZE=$bs"
            fi
        else
            echo "$sp,$bs,0,$STATUS" >> $RESULTS_FILE
        fi

        sleep 3 
    done
done

rm -f $LOG_FILE

echo "======================================================"
echo "🏆 Auto-Tune Completed!"
echo "🥇 Best Configuration : $best_config"
echo "🚀 Max tok/sec        : $best_tok_sec"
echo "======================================================"
echo "Detailed results saved to $RESULTS_FILE"
column -s, -t $RESULTS_FILE
