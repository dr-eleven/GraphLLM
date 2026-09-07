# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GraphSSR Stage 1 / GRPO 完整执行流程
====================================

一、这个文件在整个工程中的位置
------------------------------

`run_rl_stage1.sh` 通过下面的模块入口启动训练：

    python3 -m verl.trainer.main_ppo --config-name _generated_ppo_trainer ...

真正运行的是原始 `main_ppo.py`。本文件 `main_ppo_note.py` 是它的中文注释副本，
用于理解代码，不替换原始入口。它保留了相同的函数、类和执行结构，也可以单独通过
Python 语法检查。

入口文件本身主要负责“组装训练系统”，不直接实现每一步 GRPO 数学计算：

    main_ppo.py
        ├── Hydra：读取 YAML，并合并 stage1.sh 传入的命令行覆盖项
        ├── Ray：启动集群运行时和一个远程 TaskRunner
        ├── TaskRunner：选择 Worker 类型、数据集、奖励函数和 GPU 资源池
        ├── RayPPOTrainer.init_workers()：在 GPU 上创建实际 WorkerGroup
        └── RayPPOTrainer.fit()：进入真正的 GRPO 训练循环

二、从 Shell 到训练循环的调用链
----------------------------

1. `run_rl_stage1.sh`
   - 清理旧 Ray 进程并设置 CUDA/vLLM 环境变量。
   - 指定 SFT 模型、训练/验证 parquet、输出目录。
   - 使用 `_generated_ppo_trainer` 配置作为基础配置。
   - 用命令行参数覆盖 GRPO、batch、FSDP、vLLM、KL 和 checkpoint 配置。

2. Hydra 调用 `main(config)`
   - Hydra 先合并基础 YAML、插值字段和 Shell 覆盖项。
   - `main()` 本身只把最终配置传给 `run_ppo(config)`。

3. `run_ppo(config)` 初始化 Ray
   - 读取 verl 默认 Ray runtime_env。
   - 合并用户配置中的 Ray 环境变量。
   - 调用 `ray.init()` 启动或连接 Ray。
   - 创建一个只占 1 个 CPU 的远程 `TaskRunner` actor。
   - Driver 用 `ray.get()` 等待 `TaskRunner.run(config)` 完成。

4. `TaskRunner.run(config)` 组装训练组件
   - 根据 actor strategy 选择 FSDP/FSDP2 或 Megatron Worker。
   - 注册 ActorRollout、Critic、RefPolicy、RewardModel 等角色。
   - 校验配置之间的依赖关系和 batch 可整除关系。
   - 将模型路径复制/解析为本地路径。
   - 创建 tokenizer；多模态模型还会创建 processor。
   - 创建训练 reward manager 和验证 reward manager。
   - 创建 Ray GPU 资源池。
   - 创建训练/验证数据集和 sampler。
   - 构造 `RayPPOTrainer`，然后依次调用：

         trainer.init_workers()
         trainer.fit()

5. `RayPPOTrainer.init_workers()` 创建分布式 Worker
   - 根据 `trainer.n_gpus_per_node=3` 和 `trainer.nnodes=1` 建立三卡资源池。
   - 创建 ActorRollout WorkerGroup：FSDP actor 与 vLLM rollout 共置。
   - GRPO 不需要 value critic，因此 critic 类虽然在入口阶段完成注册，
     `need_critic(config)` 为 False 时不会创建和参与训练。
   - 因 `actor.use_kl_loss=True`，创建 reference policy WorkerGroup。
   - Stage 1 已将 reference 设置为 BF16 且 `param_offload=False`，所以 reference
     FSDP 分片常驻 GPU，不触发受容器 memlock 限制的 CPU pinned-memory 路径。
   - 初始化 vLLM、actor、reference、optimizer 和 checkpoint manager。

6. `RayPPOTrainer.fit()` 执行训练
   - 先自动查找并恢复 checkpoint；当前目录存在 `global_step_3` 时从 step 3 继续。
   - 默认在训练前跑一次完整验证。
   - 外层遍历 epoch，内层遍历 train dataloader。
   - 每个训练 step 的核心数据流如下：

         原始 prompt batch（train_batch_size=24）
             │
             ├── 每个 prompt 重复 rollout.n=3 次
             │
             ▼
         vLLM generate_sequences() 生成 72 条 response
             │
             ▼
         reward_fn 根据格式、答案和任务规则计算 token/sequence reward
             │
             ├── actor.compute_log_prob() 计算 old_log_probs
             ├── reference.compute_ref_log_prob() 计算 ref_log_probs
             └── GRPO 按同一 prompt 的 3 条 response 分组计算相对 advantage
             │
             ▼
         actor.update_actor()
             ├── PPO clipped policy loss
             ├── actor KL loss（系数 0.001）
             ├── backward
             └── optimizer step
             │
             ▼
         记录 metrics → 按 save_freq 保存 checkpoint → 进入下一 step

三、Stage 1 当前配置对应的实际算法分支
---------------------------------

- `algorithm.adv_estimator=grpo`
  使用组相对优势估计，不训练 value critic。

- `rollout.n=3`
  每个 prompt 生成 3 个回答；GRPO 在同一 prompt 的回答组内比较 reward。

- `algorithm.use_kl_in_reward=False`
  不从 reward 中扣除 KL 惩罚，reward 保持任务奖励本身。

- `actor.use_kl_loss=True`
  actor 更新时仍然使用 reference log-prob 计算 KL loss。
  所以 reference policy 仍然必需，不能因为没有 reward KL 就删除。

- `actor.fsdp_config.param_offload=True`
  actor 在 rollout/训练阶段之间手动上下 GPU，以便和 vLLM 共置并节约显存。

- `ref.fsdp_config.param_offload=False`
  reference BF16 FSDP 分片常驻 GPU，绕过 FSDP1 CPUOffload 的 `pin_memory()`。

- `actor.use_dynamic_bsz=True`
  actor 根据 token 数动态组织 micro batch；`ppo_max_token_len_per_gpu=4096`
  是单卡每个动态 micro batch 的 token 上限。

- `trainer.test_freq=-1`
  训练前仍会验证，但训练过程中不做周期验证。

- `trainer.save_freq=50`
  每 50 step、最后一步或弹性实例即将到期时保存 checkpoint。

四、几个容易混淆的边界
--------------------

1. 本文件不实现 vLLM 生成；生成逻辑在 ActorRollout Worker 中。
2. 本文件不实现 GRPO advantage；它由 `ray_trainer.py` 中的
   `compute_advantage()` 调度算法模块完成。
3. 本文件不实现 reward 细节；`load_reward_manager()` 会根据配置加载具体实现。
4. 本文件不执行 FSDP forward/backward；这些操作发生在远程 GPU Worker 中。
5. `ray.get()` 看似只有一行，但它会阻塞 Driver，直到整个远程训练任务结束或报错。

下面开始逐行解释入口代码。
"""

# 导入 os，用于读取当前远程 TaskRunner 进程的 PID。
import os
# 导入 socket，用于打印运行 TaskRunner 的节点主机名。
import socket

# 导入 Hydra；它负责加载 YAML 配置并应用命令行覆盖项。
import hydra
# 导入 Ray；它负责远程 actor、GPU Worker 和跨进程 RPC。
import ray
# 导入 OmegaConf；Hydra 的配置对象以它为底层表示。
from omegaconf import OmegaConf

# 导入 verl 的抽象 sampler，用于校验外部 curriculum sampler 类型。
from verl.experimental.dataset.sampler import AbstractSampler
# 导入 PPO 默认 Ray runtime_env，其中包含 NCCL、vLLM 等环境变量。
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
# 导入分布式 PPO/GRPO Trainer；真正训练循环位于这个类的 fit()。
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
# 导入 reward manager 加载器；它会按配置构造训练和验证奖励函数。
from verl.trainer.ppo.reward import load_reward_manager
# 导入两个分支判断函数：是否需要 critic、是否需要 reference policy。
from verl.trainer.ppo.utils import need_critic, need_reference_policy
# 导入配置校验函数，用于在分配大量 GPU 资源前尽早发现不合法配置。
from verl.utils.config import validate_config
# 导入 CUDA 可用性标志，决定是否可以启用 CUDA Nsight profiling。
from verl.utils.device import is_cuda_available
# 导入动态类型加载器，用于加载用户自定义 Dataset 或 Sampler。
from verl.utils.import_utils import load_extern_type


# 注册 Hydra 入口：默认读取 trainer/config/ppo_trainer.yaml。
# Stage 1 Shell 通过 `--config-name _generated_ppo_trainer` 覆盖默认 config_name。
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
# 定义程序主入口；Hydra 会把合并完成的 DictConfig 注入 config 参数。
def main(config):
    """PPO/GRPO 的 Hydra 主入口。

    参数:
        config: Hydra 合并基础 YAML、插值和命令行覆盖项后的最终配置。
    """
    # 将配置交给 Ray 初始化层；main 本身不承担训练逻辑。
    run_ppo(config)


# 定义启动 Ray 并运行远程训练任务的函数。
def run_ppo(config, task_runner_class=None) -> None:
    """初始化 Ray，并在远程 TaskRunner 中运行分布式 PPO/GRPO。

    参数:
        config: 完整训练配置，包括模型、数据、Ray、GPU 和算法参数。
        task_runner_class: recipe 可注入自定义 TaskRunner；None 时使用本文件实现。
    """
    # 只在当前 Driver 尚未连接 Ray 时初始化，避免重复 ray.init()。
    if not ray.is_initialized():
        # 取得 verl 为 PPO 预设的 Ray runtime_env。
        default_runtime_env = get_ppo_ray_runtime_env()
        # 读取用户提供的 ray.init 参数；缺省时使用空字典。
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        # 从 ray.init 参数中单独取出 runtime_env，便于和默认值合并。
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        # 如果启用了 transfer queue，需要把开关传播给所有 Ray Worker。
        if config.transfer_queue.enable:
            # 取得用户 runtime_env 中已有的环境变量。
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            # 写入 transfer queue 的进程环境开关。
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            # 将更新后的环境变量放回 runtime_env。
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        # 合并默认 runtime_env 和用户 runtime_env；用户值具有覆盖作用。
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        # 创建新的 OmegaConf，把合并后的 runtime_env 放回 ray.init 参数。
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        # 打印最终 Ray 初始化参数，便于从日志检查环境变量是否正确传播。
        print(f"ray init kwargs: {ray_init_kwargs}")
        # 转成普通 Python 容器后展开参数，启动或连接 Ray 集群。
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # recipe 没有传入自定义 TaskRunner 时，包装本文件的 TaskRunner。
    if task_runner_class is None:
        # 声明远程 TaskRunner 只占 1 个 CPU；实际 GPU 由后续资源池管理。
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)

    # 判断是否为 CUDA 平台上的 Nsight Systems 指定 step profiling。
    if (
        # 只有 CUDA 平台支持这里的 nsys 分支。
        is_cuda_available
        # profiling 工具必须明确配置为 nsys。
        and config.global_profiler.tool == "nsys"
        # steps 不能是 None。
        and config.global_profiler.get("steps") is not None
        # steps 还必须至少包含一个训练 step。
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        # 延迟导入 NVTX 检查函数，普通训练不会承担这部分依赖。
        from verl.utils.import_utils import is_nvtx_available

        # Nsight 标记依赖 nvtx 包；缺失时给出明确安装提示。
        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        # 将 OmegaConf 的 Nsight controller 参数转成 Ray 可序列化的普通对象。
        nsight_options = OmegaConf.to_container(
            # 读取 controller 侧的 Nsight Systems 参数。
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        # 为 TaskRunner 注入 Nsight runtime_env，然后创建远程实例。
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    # 普通 Stage 1 不启用 nsys，直接创建远程 TaskRunner。
    else:
        # 创建 TaskRunner Ray actor 实例。
        runner = task_runner_class.remote()
    # 远程调用 TaskRunner.run(config)，并阻塞等待整个训练完成或抛出异常。
    ray.get(runner.run.remote(config))

    # 读取可选的 Ray timeline 输出路径。
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    # 只有配置了路径才导出 timeline，避免普通训练产生额外文件。
    if timeline_json_file:
        # 将 Ray task/actor 时间线写成 Chrome trace JSON。
        ray.timeline(filename=timeline_json_file)


# 定义远程训练编排器；run_ppo() 会把这个类包装成 Ray actor。
class TaskRunner:
    """在 Ray 远程进程中组装并执行 PPO/GRPO 训练。

    属性:
        role_worker_mapping: Role 到 Ray Worker 类的映射。
        mapping: Role 到资源池 ID 的映射。
    """

    # 初始化一个尚未注册任何 Worker 的 TaskRunner。
    def __init__(self):
        # 保存 ActorRollout、Critic、RefPolicy、RewardModel 对应的 Worker 类。
        self.role_worker_mapping = {}
        # 保存每个 Role 应该使用哪个 Ray GPU 资源池。
        self.mapping = {}

    # 根据 actor strategy 注册 actor+rollout Worker。
    def add_actor_rollout_worker(self, config):
        """按 actor strategy 注册 ActorRollout Worker。"""
        # 导入默认 Ray WorkerGroup；它负责把多个远程 Worker 组织成一组。
        from verl.single_controller.ray import RayWorkerGroup

        # FSDP 和 FSDP2 使用 fsdp_workers 中的 Worker 实现。
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            # 同时导入同步和异步 rollout Worker 类。
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            # 根据 rollout.mode 选择异步或同步实现。
            actor_rollout_cls = (
                # async 模式使用异步 Worker。
                AsyncActorRolloutRefWorker
                # Stage 1 默认 sync，因此实际选择 ActorRolloutRefWorker。
                if config.actor_rollout_ref.rollout.mode == "async"
                # 非 async 模式使用同步 Worker。
                else ActorRolloutRefWorker
            )
            # FSDP 路径使用默认 RayWorkerGroup。
            ray_worker_group_cls = RayWorkerGroup

        # Megatron strategy 使用 megatron_workers 中的实现。
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            # 导入 Megatron 同步和异步 Worker。
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

            # 同样依据 rollout.mode 选择具体 Worker 类。
            actor_rollout_cls = (
                # async 模式选择 Megatron 异步 Worker。
                AsyncActorRolloutRefWorker
                # 检查 rollout.mode。
                if config.actor_rollout_ref.rollout.mode == "async"
                # sync 模式选择 Megatron 同步 Worker。
                else ActorRolloutRefWorker
            )
            # Megatron 路径目前也使用默认 RayWorkerGroup。
            ray_worker_group_cls = RayWorkerGroup

        # 其他 actor strategy 尚未实现。
        else:
            # 立即报错，避免后续以不完整 Worker 配置启动训练。
            raise NotImplementedError

        # 延迟导入 Role 枚举，避免模块加载阶段出现不必要的循环依赖。
        from verl.trainer.ppo.ray_trainer import Role

        # 将选中的 Worker 类包装成 Ray remote class，并注册为 ActorRollout。
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        # 返回 Worker 类和 WorkerGroup 类，供 reference 注册及 Trainer 构造使用。
        return actor_rollout_cls, ray_worker_group_cls

    # 注册 critic Worker 类；是否实际创建由 need_critic/config 决定。
    def add_critic_worker(self, config):
        """将 critic Worker 类加入 Role 映射。"""
        # FSDP/FSDP2 critic 根据 legacy 开关选择旧或新 Worker 架构。
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            # auto/enable 使用旧实现，disable 使用新 roles 实现。
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            # Stage 1 默认 auto，进入旧 FSDP Worker 分支。
            if use_legacy_worker_impl in ["auto", "enable"]:
                # 导入旧版 FSDP CriticWorker。
                from verl.workers.fsdp_workers import CriticWorker
            # 显式 disable 时使用新 Worker 架构。
            elif use_legacy_worker_impl == "disable":
                # 导入新版 roles.CriticWorker。
                from verl.workers.roles import CriticWorker

                # 在日志中提示已切换到新 Worker 实现。
                print("Using new worker implementation")
            # 拒绝拼写错误或未知开关值。
            else:
                # 抛出包含实际配置值的错误。
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        # Megatron critic 使用 megatron_workers 实现。
        elif config.critic.strategy == "megatron":
            # 导入 Megatron CriticWorker。
            from verl.workers.megatron_workers import CriticWorker

        # 其他 critic strategy 未实现。
        else:
            # 在资源分配前终止。
            raise NotImplementedError

        # 导入 Role 枚举。
        from verl.trainer.ppo.ray_trainer import Role

        # 把 CriticWorker 包装成 Ray remote class 并登记。
        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    # 创建 Role 到 GPU 资源池的布局。
    def init_resource_pool_mgr(self, config):
        """创建 Ray GPU 资源池管理器。"""
        # 导入 Role 枚举。
        from verl.trainer.ppo.ray_trainer import Role

        # 定义主资源池名称。
        global_pool_id = "global_pool"
        # 每个节点提供 n_gpus_per_node 个 GPU，共创建 nnodes 个节点槽位。
        resource_pool_spec = {
            # Stage 1 得到 {"global_pool": [3]}。
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        # 奖励模型可以选择使用独立资源池。
        if config.reward_model.enable_resource_pool:
            # 独立 reward pool 的单节点 GPU 数必须为正数。
            if config.reward_model.n_gpus_per_node <= 0:
                # 非法时给出准确字段名。
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            # 独立 reward pool 的节点数也必须为正数。
            if config.reward_model.nnodes <= 0:
                # 非法时终止。
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            # 构造 reward pool 的每节点 GPU 列表。
            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            # 将独立 reward pool 加入资源规格。
            resource_pool_spec["reward_pool"] = reward_pool

        # ActorRollout 始终映射到主 GPU 池。
        self.mapping[Role.ActorRollout] = global_pool_id
        # Critic 也映射到主池；GRPO 下不会真正实例化 critic。
        self.mapping[Role.Critic] = global_pool_id
        # 导入资源池管理器。
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        # 用资源规格和 Role 映射创建管理器。
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        # 返回给 RayPPOTrainer。
        return resource_pool_manager

    # 按配置注册可选的模型型 RewardModel Worker。
    def add_reward_model_worker(self, config):
        """仅在 reward_model.enable=True 时注册 RewardModel Worker。"""
        # 导入 Role 枚举。
        from verl.trainer.ppo.ray_trainer import Role

        # 只有模型型奖励启用时才选择和注册 Worker。
        if config.reward_model.enable:
            # 读取 legacy/new Worker 架构开关。
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            # auto/enable 使用旧版 Worker。
            if use_legacy_worker_impl in ["auto", "enable"]:
                # FSDP/FSDP2 奖励模型使用 fsdp_workers。
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    # 导入 FSDP RewardModelWorker。
                    from verl.workers.fsdp_workers import RewardModelWorker
                # Megatron 奖励模型使用 megatron_workers。
                elif config.reward_model.strategy == "megatron":
                    # 导入 Megatron RewardModelWorker。
                    from verl.workers.megatron_workers import RewardModelWorker
                # 其他 reward strategy 未实现。
                else:
                    # 终止初始化。
                    raise NotImplementedError
            # disable legacy 时使用新 roles Worker。
            elif use_legacy_worker_impl == "disable":
                # 导入新版 RewardModelWorker。
                from verl.workers.roles import RewardModelWorker

                # 输出实现切换提示。
                print("Using new worker implementation")
            # 拒绝未知开关值。
            else:
                # 抛出带实际值的异常。
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            # 把奖励模型类包装成 Ray remote class。
            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            # 独立 reward pool 启用时将角色映射过去。
            if config.reward_model.enable_resource_pool:
                # RewardModel 使用独立 GPU 池。
                self.mapping[Role.RewardModel] = "reward_pool"
            # 否则与 actor 等角色共用 global_pool。
            else:
                # RewardModel 使用主 GPU 池。
                self.mapping[Role.RewardModel] = "global_pool"

    # 根据 KL 配置注册 reference policy。
    def add_ref_policy_worker(self, config, ref_policy_cls):
        """KL reward 或 actor KL loss 启用时注册 reference policy。"""
        # 导入 Role 枚举。
        from verl.trainer.ppo.ray_trainer import Role

        # reward KL 与 actor KL loss 任一启用都需要 reference log-prob。
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            # reference 复用 actor 对应的 Worker 类，但以 RefPolicy role 初始化。
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            # reference 与 ActorRollout 共用 global_pool 中的 GPU。
            self.mapping[Role.RefPolicy] = "global_pool"

    # TaskRunner 的核心远程方法：组装并启动 Trainer。
    def run(self, config):
        """执行训练系统的完整装配流程。"""
        # 延迟导入 pprint，使配置以更易读的形式写入日志。
        from pprint import pprint

        # 在远程进程中重新导入 OmegaConf。
        from omegaconf import OmegaConf

        # 导入模型路径本地化工具，支持 HDFS、共享内存或本地路径。
        from verl.utils.fs import copy_to_local

        # 打印远程 TaskRunner 所在机器和 PID，便于排查 Ray 调度问题。
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        # 解析配置插值并转成普通容器后完整打印。
        pprint(OmegaConf.to_container(config, resolve=True))
        # 原地解析 OmegaConf 中尚未求值的 `${...}` 插值。
        OmegaConf.resolve(config)

        # 注册 ActorRollout Worker，并取得 reference 可复用的 Worker 类。
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        # 注册 Critic 类；GRPO 下 Trainer 的 use_critic=False，不会实例化它。
        self.add_critic_worker(config)

        # 按 reward_model.enable 决定是否注册模型型 RewardModel Worker。
        # GraphSSR 主要由 reward manager 执行规则/任务奖励。
        self.add_reward_model_worker(config)

        # Stage 1 的 actor.use_kl_loss=True，因此这里会注册 reference。
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # 在启动大模型 Worker 前校验所有配置依赖与 batch 关系。
        validate_config(
            # 传入完整 Hydra 配置。
            config=config,
            # 根据 Role 映射判断 reference 是否启用。
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            # GRPO 会在这里判定为不需要 critic。
            use_critic=need_critic(config),
        )

        # 将 actor 模型 checkpoint 解析/复制到当前节点的本地路径。
        local_path = copy_to_local(
            # Stage 1 指向 SFT 输出 graphssr_sft。
            config.actor_rollout_ref.model.path,
            # use_shm=False 时不复制到 /dev/shm。
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        # 导入 Hugging Face processor/tokenizer 构造工具。
        from verl.utils import hf_processor, hf_tokenizer

        # 读取是否信任模型仓库中的自定义 Python 代码。
        trust_remote_code = config.data.get("trust_remote_code", False)
        # 从本地模型目录创建 tokenizer。
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # 为多模态模型创建 processor；纯文本模型通常返回 None。
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # 创建训练 reward manager；num_examine=0 表示默认不打印样本详情。
        reward_fn = load_reward_manager(
            # 传入完整配置和 tokenizer。
            config,
            tokenizer,
            # 训练阶段不额外展示检查样本。
            num_examine=0,
            # 展开可选 reward_kwargs。
            **config.reward_model.get("reward_kwargs", {}),
        )
        # 创建验证 reward manager；num_examine=1 会展示一个验证样本便于检查。
        val_reward_fn = load_reward_manager(
            # 验证使用同一套配置。
            config,
            tokenizer,
            # 检查一个样本。
            num_examine=1,
            # 展开相同的 reward_kwargs。
            **config.reward_model.get("reward_kwargs", {}),
        )

        # 根据三卡单节点配置创建 Ray GPU 资源池管理器。
        resource_pool_manager = self.init_resource_pool_mgr(config)

        # 导入 RLHFDataset 对应的 batch collate 函数。
        from verl.utils.dataset.rl_dataset import collate_fn

        # 从训练 parquet 创建训练数据集。
        train_dataset = create_rl_dataset(
            # Stage 1 的 train_9799_graphr2_subgraph.parquet。
            config.data.train_files,
            # 数据预处理和长度限制配置。
            config.data,
            # prompt tokenization 使用的 tokenizer。
            tokenizer,
            # 多模态 processor；文本模型可为 None。
            processor,
            # 标记为训练集，允许 DynamicGenDataset 分支。
            is_train=True,
            # -1 表示不额外限制最大样本数。
            max_samples=config.data.get("train_max_samples", -1),
        )
        # 从验证 parquet 创建验证数据集。
        val_dataset = create_rl_dataset(
            # Stage 1 的 test_600_graphr2_subgraph.parquet。
            config.data.val_files,
            # 与训练集共享数据配置。
            config.data,
            # 使用相同 tokenizer。
            tokenizer,
            # 使用相同 processor。
            processor,
            # 标记为验证集，禁用仅训练使用的动态生成分支。
            is_train=False,
            # -1 表示使用过滤后全部验证样本。
            max_samples=config.data.get("val_max_samples", -1),
        )
        # 按 shuffle/seed/custom sampler 配置创建训练 sampler。
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # 构造 RayPPOTrainer；它保存数据流所需的所有组件。
        trainer = RayPPOTrainer(
            # 完整算法和运行配置。
            config=config,
            # 文本编解码器。
            tokenizer=tokenizer,
            # 可选多模态处理器。
            processor=processor,
            # Role 到远程 Worker 类的映射。
            role_worker_mapping=self.role_worker_mapping,
            # Role 到三卡资源池的映射与管理器。
            resource_pool_manager=resource_pool_manager,
            # WorkerGroup 实现。
            ray_worker_group_cls=ray_worker_group_cls,
            # 训练奖励函数。
            reward_fn=reward_fn,
            # 验证奖励函数。
            val_reward_fn=val_reward_fn,
            # 训练数据集。
            train_dataset=train_dataset,
            # 验证数据集。
            val_dataset=val_dataset,
            # DataLoader 拼 batch 的函数。
            collate_fn=collate_fn,
            # 支持随机、顺序或 curriculum 的 sampler。
            train_sampler=train_sampler,
        )
        # 真正在 Ray 资源池中创建 actor/rollout/reference 等 GPU Worker。
        trainer.init_workers()

        # 恢复 checkpoint、执行验证并进入 GRPO step 循环。
        trainer.fit()


# 定义训练/验证数据集工厂。
def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """根据配置创建默认、外部自定义或动态生成数据集。

    参数:
        data_paths: parquet 等数据文件路径。
        data_config: 数据预处理配置。
        tokenizer: Hugging Face tokenizer。
        processor: 可选多模态 processor。
        is_train: 是否为训练集。
        max_samples: 最大样本数，-1 表示无限制。
    """
    # 导入 PyTorch Dataset 基类，用于校验外部自定义类。
    from torch.utils.data import Dataset

    # 导入 verl 默认 RLHF 数据集。
    from verl.utils.dataset.rl_dataset import RLHFDataset

    # 如果同时配置了自定义类字段和 Python 文件路径，则使用外部数据集类。
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # 按文件路径和类名动态导入 Dataset。
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # 外部类必须继承 torch.utils.data.Dataset。
        if not issubclass(dataset_cls, Dataset):
            # 类型错误时同时报告类名和来源路径。
            raise TypeError(
                # 错误消息第一部分说明类名。
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                # 错误消息第二部分说明路径和继承要求。
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    # 训练集配置了 datagen 时使用动态生成数据集。
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # 延迟导入 DynamicGenDataset。
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        # 选择动态生成类。
        dataset_cls = DynamicGenDataset
        # 在日志中提示数据不是普通静态 parquet 流程。
        print("Using DynamicGenDataset for data generation.")
    # 没有扩展配置时使用默认 RLHFDataset。
    else:
        # Stage 1 实际进入这个分支。
        dataset_cls = RLHFDataset
    # 打印最终数据集类名；日志中可见 `Using dataset class: RLHFDataset`。
    print(f"Using dataset class: {dataset_cls.__name__}")

    # 使用统一构造参数实例化选中的数据集类。
    dataset = dataset_cls(
        # 一个或多个数据文件。
        data_files=data_paths,
        # tokenizer 负责模板编码和长度检查。
        tokenizer=tokenizer,
        # processor 负责可选的图像/视频输入。
        processor=processor,
        # 传入完整 data 配置。
        config=data_config,
        # 应用可选样本上限。
        max_samples=max_samples,
    )

    # 将数据集返回给 RayPPOTrainer。
    return dataset


# 定义训练 sampler 工厂。
def create_rl_sampler(data_config, dataset):
    """创建 curriculum、随机或顺序 sampler。

    参数:
        data_config: 数据与 sampler 配置。
        dataset: 已创建的训练数据集。
    """
    # 导入 torch，以便创建独立随机数生成器。
    import torch
    # 导入 PyTorch 随机和顺序 sampler。
    from torch.utils.data import RandomSampler, SequentialSampler

    # 优先检查是否配置了外部 curriculum sampler。
    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        # 从外部模块路径加载 curriculum sampler 类。
        curriculum_class = load_extern_type(
            # Python 文件或模块路径。
            data_config.sampler.class_path,
            # 目标类名。
            data_config.sampler.class_name,
        )
        # 实例化 curriculum sampler。
        sampler = curriculum_class(
            # PyTorch sampler 的标准数据源参数。
            data_source=dataset,
            # 额外传入 data 配置。
            data_config=data_config,
        )
        # 确保自定义对象遵循 verl 的 AbstractSampler 协议。
        assert isinstance(sampler, AbstractSampler)
        # Curriculum 要在每个 batch 后重排数据，因此 DataLoader 不能预取缓存。
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            # 第一段解释约束。
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            # 第二段解释预取为何破坏 curriculum。
            "If the dataloader caches data before the batch is done the "
            # 第三段解释 sampler 会失去重排机会。
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # 没有 curriculum 且 shuffle=True 时使用可恢复的随机 sampler。
    elif data_config.shuffle:
        # 创建独立 Generator，避免依赖全局 torch RNG。
        train_dataloader_generator = torch.Generator()
        # 读取可选固定随机种子。
        seed = data_config.get("seed")
        # 只有显式配置 seed 才手动设种子。
        if seed is not None:
            # 固定 sampler 的随机序列，改善复现和 checkpoint 恢复一致性。
            train_dataloader_generator.manual_seed(seed)
        # 创建随机 sampler。
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    # shuffle=False 时按数据集顺序读取。
    else:
        # 创建顺序 sampler。
        sampler = SequentialSampler(data_source=dataset)

    # 返回给 RayPPOTrainer 构造 DataLoader。
    return sampler


# 只有作为脚本/模块直接执行时才启动 Hydra；被 import 时不会自动训练。
if __name__ == "__main__":
    # 进入 Hydra main；命令行参数会在调用函数前被解析。
    main()
