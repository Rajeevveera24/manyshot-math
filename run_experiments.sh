#!/bin/bash

if [[ "$1" != "" ]]; then
	model_name="$1"
else
	model_name='meta-llama/Llama-3.1-8B-Instruct'
    echo "Using default model: $model_name"
fi

PORT=$((RANDOM % 40000 + 8000))
echo "Using port: $PORT"

sbatch --job-name="vllm_server_${PORT}" deploy_vllm.sh $model_name $PORT

# Poll to check if the VLLM server has started
while true; do
    if squeue --me --name="vllm_server_${PORT}" -h -o %T | grep -q R; then
        echo "VLLM server job has started"
        break
    fi
    sleep 10
done

# Wait for the VLLM server to start
sleep 200

# Get hostname of the VLLM server
VLLM_HOSTNAME=$(squeue --me --name="vllm_server_${PORT}" -h -o %N)

echo "VLLM server hostname: $VLLM_HOSTNAME"

sbatch run_eval.sh $model_name $VLLM_HOSTNAME $PORT

# Wait until the above sbatch job is completed
while true; do
    if squeue --me --name=math_eval -h -o %T; then
        sleep 300 
    else
        break
    fi
done

# Cleanup: Kill the VLLM server
VLLM_PID=$(squeue --me --name="vllm_server_${PORT}" -h -o %i)
if [ ! -z "$VLLM_PID" ]; then
    scancel $VLLM_PID
fi