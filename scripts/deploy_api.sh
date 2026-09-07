#!/bin/bash

# =============================================================================
# GraphSSR 模型服务部署脚本 (deploy_api.sh)
# =============================================================================
#
# 功能概述：
#   通过 SSH 远程登录到 GPU 节点，在 Docker 容器内启动 vLLM 推理服务，
#   将模型封装为 OpenAI 兼容的 HTTP API（/v1/chat/completions）。
#
# 在训练流程中的角色：
#   基础设施层脚本，在以下三个阶段都会被调用（每次部署的模型不同）：
#     1. SFT 数据生成前  → 部署 teacher model（教师模型，生成推理轨迹）
#     2. RL 数据生成前   → 部署 SFT base model（用来评估题目难度）
#     3. 最终评测前      → 部署 final trained model（被评测的目标模型）
#
# 核心原理：
#   脚本通过并行 SSH 连接到多个 GPU 节点，在每个节点上：
#     1. 检查/加载 Docker 镜像
#     2. 清理旧的同名容器
#     3. 启动新容器（--gpus all, --net=host, --ipc=host）
#     4. 在容器内执行 vllm serve 命令
#   启动成功后，各节点上的 vLLM 实例通过 HTTP 端口对外暴露推理接口。
#
# 使用前必须修改的配置项（见下方配置区域）：
#   - NODES：GPU 节点主机名/IP 列表
#   - IMAGE_NAME / TAR_PATH：Docker 镜像名和 tar 包路径
#   - VOLUMES：宿主机到容器的目录挂载映射（模型文件必须能被容器访问）
#   - MODEL_PATH：模型 checkpoint 在容器内的路径（须与 VOLUMES 对应）
#   - TP_SIZE：Tensor Parallel 并行度（通常等于单节点 GPU 数量）
#   - PORT：vLLM 服务监听的端口号
#   - SERVED_MODEL_NAME：对外暴露的模型名称（评测/数据生成代码通过此名称调用）
#
# 注意事项：
#   - 需要配置免密 SSH，否则脚本无法远程执行命令
#   - 多次部署不同模型到同一节点时，必须使用不同端口或先后执行
#   - 如果 Docker 镜像已存在则跳过 load 步骤（节省时间）
# =============================================================================

# ====================== 配置区域（必须根据实际环境修改）======================
# 目标 GPU 节点列表（主机名或 IP 地址）
NODES=(
    YOUR_NODE_0          # 第 1 个 GPU 节点
    YOUR_NODE_1          # 第 2 个 GPU 节点
    # 可按需添加更多节点，每个节点上会启动一个 vLLM 实例
)

# Docker 镜像名称与 tar 包路径
IMAGE_NAME="your_docker_image_name"        # 镜像名（如 vllm/vllm-openai:latest）
TAR_PATH="/path/to/your/docker_image.tar"  # 镜像 tar 包在宿主机上的完整路径
CONTAINER_NAME="verl"                       # 容器名称（建议保持与 RL 训练容器一致）

# 宿主机目录到容器目录的挂载映射
# 示例：-v /data/models:/models  将宿主机的 /data/models 映射到容器的 /models
# 模型文件必须放在挂载路径下，否则容器内的 vLLM 无法加载
VOLUMES="-v /your/mount/source:/your/mount/target"

# vLLM 服务配置
MODEL_PATH="/path/to/your/model/checkpoint"  # 模型 checkpoint 路径（容器内路径！）
TP_SIZE=8                                      # Tensor Parallel 大小，建议等于单节点 GPU 数
PORT=31200                                     # vLLM API 监听端口
SERVED_MODEL_NAME="eval"                       # 对外暴露的模型名称
# 注意：SERVED_MODEL_NAME 必须与调用方代码中的 model_name 一致
#   - 部署 teacher model 时建议设为 "teacher_model"
#   - 部署 diversity model 时建议设为 "diversity_model"
#   - 部署 SFT model 时建议设为 "server_0"
#   - 部署 final eval model 时建议设为 "eval"

# 拼接完整的 vLLM 启动命令
VLLM_CMD="vllm serve $MODEL_PATH --tensor-parallel-size $TP_SIZE --port $PORT --served-model-name $SERVED_MODEL_NAME"
# =============================================================================

# ---------------------------------------------------------------------------
# deploy_on_node 函数：在单个节点上完成完整的部署流程
# 参数：$1 = 目标节点的主机名或 IP
# 执行步骤：
#   1. 检查 Docker 镜像是否存在，不存在则从 tar 包加载
#   2. 删除已有的同名容器（确保干净状态）
#   3. 启动新容器（后台运行，挂载全部 GPU，使用宿主机网络）
#   4. 在容器内启动 vLLM 服务（后台执行，日志写入 /tmp/vllm.log）
# ---------------------------------------------------------------------------
deploy_on_node() {
    local node=$1
    echo "[Node $node] 正在部署 vLLM 推理服务..."

    # 通过 SSH 在远程节点上执行一系列 Docker 操作
    ssh -o StrictHostKeyChecking=no "$node" << EOF
        # ---- 步骤 1：加载 Docker 镜像（如果本地不存在） ----
        if [[ "\$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
            echo "[Node $node] 本地未找到镜像，正在从 $TAR_PATH 加载..."
            docker load -i $TAR_PATH
        else
            echo "[Node $node] Docker 镜像已存在，跳过加载。"
        fi

        # ---- 步骤 2：清理旧的同名容器 ----
        # 先尝试停止容器内的 Ray 进程（如果存在），再强制删除容器
        if [[ "\$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" != "" ]]; then
            echo "[Node $node] 发现已存在的容器 $CONTAINER_NAME，正在删除..."
            docker rm -f $CONTAINER_NAME
        fi

        # ---- 步骤 3：启动新的 Docker 容器 ----
        # --gpus all    ：允许容器使用所有 GPU
        # --ipc=host    ：共享宿主机 IPC 命名空间（vLLM 多 GPU 通信需要）
        # --net=host    ：直接使用宿主机网络栈（避免端口映射，性能更好）
        # --entrypoint  ：覆盖镜像默认入口，改为 bash（后续手动启动 vLLM）
        echo "[Node $node] 正在创建并启动新容器..."
        docker run -d \
            --name $CONTAINER_NAME \
            --gpus all \
            --ipc=host \
            --net=host \
            --entrypoint /bin/bash \
            $VOLUMES \
            $IMAGE_NAME

        # ---- 步骤 4：在容器内启动 vLLM 推理服务 ----
        # vllm serve 命令在后台执行，日志重定向到 /tmp/vllm.log
        echo "[Node $node] 正在容器内启动 vLLM 服务..."
        docker exec -d $CONTAINER_NAME bash -c "$VLLM_CMD > /tmp/vllm.log 2>&1"

        # 获取节点的 IP 地址以输出服务访问地址
        node_ip=\$(hostname -i | awk '{print \$1}')
        echo "SUCCESS: $node 部署完成，服务地址: \$node_ip:$PORT"
EOF
}

# ---------------------------------------------------------------------------
# 主流程：并行向所有节点部署服务
# ---------------------------------------------------------------------------
echo "开始并行部署 vLLM 服务到 ${#NODES[@]} 个节点..."
start_time=$(date +%s)

# 对每个节点调用 deploy_on_node，使用 & 放入后台并行执行
for node in "${NODES[@]}"; do
    deploy_on_node "$node" &
done

# wait 等待所有并行任务完成
wait

end_time=$(date +%s)
echo "-------------------------------------------"
echo "全部节点部署完成！总耗时: $((end_time - start_time)) 秒。"

# 提示查看日志的命令
if [ ${#NODES[@]} -gt 0 ]; then
    echo "如需查看 vLLM 运行日志，请执行:"
    echo "ssh ${NODES[0]} \"docker exec $CONTAINER_NAME tail -f /tmp/vllm.log\""
fi
