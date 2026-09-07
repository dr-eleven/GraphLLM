#!/bin/bash

# =============================================================================
# GraphSSR Ray 分布式集群启动脚本 (start_ray_cluster.sh)
# =============================================================================
#
# 功能概述：
#   在多个 GPU 节点上启动 Ray 分布式计算集群，为后续的强化学习（RL）训练
#   提供分布式执行环境。Ray 负责跨节点的任务调度、资源分配和进程间通信。
#
# 在训练流程中的角色：
#   RL 基础设施层脚本，必须在 RL Stage 1（SFT 策略训练）和 Stage 2（RL 策略
#   优化）之前执行。只有 Ray 集群正常启动后，verl 框架才能将 PPO 训练任务
#   分发到多个 GPU 节点上并行执行。
#
# 核心原理：
#   1. 在每个节点上通过 SSH 创建/重建 Docker 容器（统一运行环境）
#   2. 在第一个节点（Head Node）上启动 Ray Head 进程（调度中心 + GCS）
#   3. 在其余节点上启动 Ray Worker 进程，连接到 Head 节点
#   4. 通过 Python 脚本验证所有节点是否成功注册为 Alive 状态
#   5. 如果节点数不足，自动清理并重试（循环直到全部节点就绪）
#
# 使用前必须修改的配置项（见下方配置区域）：
#   - NODES：所有参与集群的节点主机名/IP 列表（第一个自动成为 Head）
#   - CONTAINER_NAME：Docker 容器名称
#   - IMAGE_TAG：Docker 镜像标签
#   - IMAGE_PATH：Docker 镜像 tar 包在宿主机上的路径
#   - DOCKER_RUN_CMD 中的 -v 挂载：确保训练代码和数据可被容器访问
#
# 注意事项：
#   - 所有节点之间需要免密 SSH
#   - 容器使用 --network host 模式，Ray 直接绑定宿主机 IP
#   - Head 节点使用默认端口 6379（Ray GCS）
#   - 如果验证失败会自动重试（无限循环），确保最终一定成功或人工介入
# =============================================================================

# ====================== 配置区域（必须根据实际环境修改）======================
# 集群节点列表（第一个节点自动成为 Head 节点）
NODES=(
    YOUR_NODE_0          # Head 节点（Ray 调度中心）
    YOUR_NODE_1          # Worker 节点（执行计算任务）
    # 可按需添加更多 Worker 节点
)

# Docker 容器配置
CONTAINER_NAME="verl"                          # 容器名称
IMAGE_TAG="your_docker_image_name"             # Docker 镜像标签
IMAGE_PATH="/path/to/your/docker_image.tar"    # 镜像 tar 包路径

# Docker run 完整命令
# --network host   : 使用宿主机网络（Ray 多节点通信必需）
# --ipc host        : 共享 IPC 命名空间（PyTorch 多进程数据加载需要）
# --gpus all        : 允许容器访问所有 GPU
# --shm-size 32g    : 共享内存 32GB（多进程 dataloader 需要）
# --privileged      : 特权模式（部分集群环境需要）
# -v 挂载           : 将训练代码和数据目录映射进容器
DOCKER_RUN_CMD="docker run -d -it --network host --ipc host --name $CONTAINER_NAME --gpus all --shm-size 32g --privileged -v /your/mount/source:/your/mount/target $IMAGE_TAG"

# 自动计算目标节点总数
TARGET_COUNT=${#NODES[@]}
# 第一个节点作为 Head 节点
HEAD_NODE=${NODES[0]}
# =============================================================================

# ---------------------------------------------------------------------------
# setup_node 函数：在单个节点上初始化 Docker 容器
# 参数：$1 = 目标节点的主机名或 IP
# 执行步骤：
#   1. 检查 Docker 镜像是否存在，不存在则从 tar 包加载
#   2. 停止已有的 Ray 进程（避免端口冲突）
#   3. 删除已有的同名容器（确保干净状态）
#   4. 启动新容器
# ---------------------------------------------------------------------------
setup_node() {
    local node=$1
    echo "[Node: $node] 正在初始化容器..."

    # ---- 步骤 1：检查并加载 Docker 镜像 ----
    # 如果镜像不存在于本地，从 tar 文件加载
    if ! ssh $node "docker images -q $IMAGE_TAG" | grep -q . ; then
        ssh $node "docker load -i $IMAGE_PATH"
    fi

    # ---- 步骤 2 & 3：清理旧进程和容器 ----
    # ray stop：停止该节点上容器内可能残留的 Ray 进程
    # docker rm -f：强制删除旧容器（如果存在）
    # 错误输出重定向到 /dev/null（首次运行时这些命令会失败，属于正常情况）
    ssh $node "docker exec $CONTAINER_NAME ray stop" &>/dev/null
    ssh $node "docker rm -f $CONTAINER_NAME" &>/dev/null

    # ---- 步骤 4：启动新容器 ----
    # 使用预定义的 DOCKER_RUN_CMD 在后台启动
    ssh $node "$DOCKER_RUN_CMD"
}

# ---------------------------------------------------------------------------
# 主流程：循环尝试启动集群直到所有节点就绪
# 采用无限重试机制，因为分布式环境可能出现临时网络波动或资源竞争
# ---------------------------------------------------------------------------
ATTEMPT=1
while true; do
    echo "================================================"
    echo "尝试启动集群 (第 $ATTEMPT 次尝试) - 目标节点数: $TARGET_COUNT"
    echo "================================================"

    # 获取 Head 节点的 IP 地址（Worker 节点需要通过此 IP 连接）
    HEAD_IP=$(ssh $HEAD_NODE "hostname -I | awk '{print \$1}'")

    # ---------- 阶段 1：并行重建所有节点的容器 ----------
    # 在所有节点上同时执行 setup_node，通过 & 后台并行 + wait 等待全部完成
    echo "--- 阶段 1: 重建节点容器 ---"
    for node in "${NODES[@]}"; do
        setup_node "$node" &
    done
    wait

    # ---------- 阶段 2：启动 Ray Head 节点 ----------
    # --head              ：将此节点初始化为 Head（包含 GCS 全局调度器）
    # --port=6379         ：GCS 服务端口
    # --disable-usage-stats：禁用 Ray 使用统计上报
    echo "--- 阶段 2: 启动 Ray Head ---"
    ssh $HEAD_NODE "docker exec $CONTAINER_NAME ray start --head --port=6379 --disable-usage-stats"
    # 等待 5 秒确保 GCS 服务完全启动，Worker 才能成功注册
    sleep 5

    # ---------- 阶段 3：并行启动所有 Worker 节点 ----------
    # --address=$HEAD_IP:6379：指定要连接的 Head 节点地址
    echo "--- 阶段 3: 启动 Ray Workers ---"
    for node in "${NODES[@]:1}"; do
        ssh $node "docker exec $CONTAINER_NAME ray start --address=$HEAD_IP:6379 --disable-usage-stats" &
    done
    wait

    # ---------- 阶段 4：验证集群状态 ----------
    # 通过 Python 脚本在 Head 节点容器内查询 Ray 的活跃节点数
    # 使用 ray.nodes() 获取节点列表，过滤出 Alive=True 的节点
    echo "--- 阶段 4: 验证集群状态 ---"
    sleep 3  # 等待 Worker 完成注册

    # 在 Head 节点的容器内执行 Python 代码获取活跃节点数
    # 如果查询失败（如 Ray 尚未完全启动），捕获异常并返回 0
    ACTIVE_COUNT=$(ssh $HEAD_NODE "docker exec $CONTAINER_NAME python3 -c \"
import ray
import sys
try:
    # 尝试连接本地 Ray 实例（address='auto' 会自动发现本地运行的 Ray）
    ray.init(address='auto', _node_ip_address='127.0.0.1')
    # 统计所有存活节点
    nodes = [n for n in ray.nodes() if n['Alive']]
    print(len(nodes))
except Exception:
    # 如果连接失败或超时，返回 0
    print(0)
\"" 2>/dev/null)

    # 防御性检查：确保 ACTIVE_COUNT 是有效数字（防止 SSH 超时返回空字符串）
    if [[ ! "$ACTIVE_COUNT" =~ ^[0-9]+$ ]]; then
        ACTIVE_COUNT=0
    fi

    echo "当前活跃节点: $ACTIVE_COUNT / $TARGET_COUNT"

    # ---------- 判断结果 ----------
    if [ "$ACTIVE_COUNT" -eq "$TARGET_COUNT" ]; then
        # 成功：所有节点都已注册且存活
        echo "✅ 集群启动成功！所有 $TARGET_COUNT 个节点已就绪。"
        # 打印完整的 Ray 集群状态（包括 GPU 资源信息）
        ssh $HEAD_NODE "docker exec $CONTAINER_NAME ray status"
        break
    else
        # 失败：节点数不足，清理所有节点的 Ray 进程后重试
        echo "❌ 启动未达预期（当前 $ACTIVE_COUNT/$TARGET_COUNT），正在清理并重试..."
        # 并行停止所有节点上的 Ray 进程
        for node in "${NODES[@]}"; do
            ssh $node "docker exec $CONTAINER_NAME ray stop" &>/dev/null &
        done
        wait
        sleep 5
        ATTEMPT=$((ATTEMPT+1))
    fi
done
