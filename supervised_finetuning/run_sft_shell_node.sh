#! /bin/bash

# GraphSSR per-node SFT worker script.
# Function:
# - Loads the Docker image, starts the SFT container, and launches LlamaFactory training on one node.
# Training role:
# - Node-level helper used by `run_sft_shell.sh` during distributed SFT execution.

ACTION="$1" # prepare, start, 或 exec
MASTER_ADDR="$2"
MASTER_PORT="$3"
DATA_DIR="$4"
EXEC_DIR="$5"
NNODES="$6"
INDEX="$7"
TASK_NAME="$8"
YAML_FILE="$9"

IMAGE_NAME="your_docker_image_name"
IMAGE_TAR="/path/to/your/docker_image.tar"
CONTAINER_NAME="sft_${TASK_NAME}"

# --- 逻辑处理 ---

if [ "$ACTION" == "prepare" ]; then
    echo "[Node $INDEX] 正在清理旧容器并检查镜像..."
    docker rm -f "${CONTAINER_NAME}" > /dev/null 2>&1
    if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
        echo "[Node $INDEX] 正在加载镜像 (可能需要几分钟)..."
        docker load -i "$IMAGE_TAR"
    fi

elif [ "$ACTION" == "start" ]; then
    echo "[Node $INDEX] 正在建立基础容器..."
    # 启动一个常驻的空容器，不直接跑训练任务
    docker run -d \
        --rm \
        --network host \
        --ipc host \
        --name "${CONTAINER_NAME}" \
        --gpus all \
        --shm-size 32G \
        -e NCCL_IB_DISABLE=0 \
        -e NCCL_SOCKET_IFNAME=ib \
        -e NCCL_IB_HCA=mlx5 \
        -e NCCL_IB_GID_INDEX=3 \
        -v /your/mount/source:/your/mount/target \
        -v ${EXEC_DIR}/dataset_info.json:/app/data/dataset_info.json \
        -v ${DATA_DIR}/graph_sft_data.json:/app/data/graph_sft_data.json \
        -v ${EXEC_DIR}/${YAML_FILE}:/app/examples/train_full/${YAML_FILE} \
        --privileged \
        ${IMAGE_NAME} \
        tail -f /dev/null

    # 循环检查容器是否真正启动成功
    SUCCESS=0
    for i in {1..10}; do
        STATUS=$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null)
        if [ "$STATUS" == "true" ]; then
            echo "[Node $INDEX] 容器已建立并处于运行状态。"
            SUCCESS=1
            break
        fi
        sleep 2
    done

    if [ $SUCCESS -eq 0 ]; then
        echo "[Node $INDEX] 错误：容器启动失败！"
        exit 1
    fi

elif [ "$ACTION" == "exec" ]; then
    echo "[Node $INDEX] 容器已就绪，开始执行分布式训练任务..."
    # 使用 docker exec 异步触发任务
    docker exec -d "${CONTAINER_NAME}" bash -c "
        FORCE_TORCHRUN=1 NNODES=${NNODES} NODE_RANK=${INDEX} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} \
        python -m llamafactory.cli train examples/train_full/${YAML_FILE}
    "
    echo "[Node $INDEX] 训练进程已拉起。"
fi
