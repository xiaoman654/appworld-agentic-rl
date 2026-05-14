# 基于 AppWorld 的低资源工具调用智能体后训练项目实行方案

## 1. 项目目标

本项目围绕 AppWorld 构建一个低资源 7B 工具调用智能体后训练与评估体系，核心目标不是单次跑通训练，而是系统比较 Base、SFT-only、SFT+GRPO 在多工具/API 交互任务中的差异，并分析 7B 模型的能力边界。

主线模型建议从 `Qwen/Qwen2.5-Coder-7B-Instruct` 开始。24GB 显存租机用于环境验证、数据处理、基座审计、小规模 QLoRA-SFT dry run；AutoDL A800 * 2 用于正式 QLoRA-SFT、批量 rollout、QLoRA-GRPO 和最终评估。Kaggle T4 * 2 可以作为备选，但需要隔离 Python 环境以避免 AppWorld 的 Pydantic v1 依赖与 Kaggle 预装包冲突。

## 2. 运行平台分工

### 24GB GPU: C0/C1/SFT Dry Run

适合任务：

- AppWorld 安装、数据下载、reference solution 验证。
- 30-50 个任务的基座模型能力审计，优先使用 4-bit 推理。
- SFT 数据构造、API 检索、任务难度标注。
- 小规模 QLoRA-SFT dry run，例如 50-200 条样本、100-300 step。
- rollout pipeline dry run，例如每个任务 K=2、max_steps=3。

推荐配置：

- Python 3.11 优先。AppWorld 当前 PyPI 要求是 `>=3.11,<4.0`，GitHub 安装示例也使用 Python 3.11。
- CUDA 12.1 或 12.4 对应的 PyTorch。
- QLoRA rank 8/16。
- `seq_len=2048` 起步，稳定后尝试 3072。
- `micro_batch=1`，`gradient_accumulation=8-16`。

### Kaggle: T4 * 2, Optional

适合任务：

- AppWorld 安装、数据下载、reference solution 验证。
- 30-50 个任务的基座模型能力审计，优先使用 4-bit 推理。
- SFT 数据构造、API 检索、任务难度标注。
- 小规模 QLoRA-SFT dry run，例如 50-200 条样本、100-300 step。
- rollout pipeline dry run，例如每个任务 K=2、max_steps=3。

不建议任务：

- 大规模 GRPO。
- 大 batch rollout。
- 高并发 vLLM server。
- 长上下文 7B 全量训练。

### AutoDL: A800 * 2

适合任务：

- 正式 QLoRA-SFT。
- vLLM 加速批量 rollout。
- QLoRA-GRPO。
- 多 reward 版本消融。
- 完整 dev/test 评估和失败分析。

推荐资源使用：

- GPU 0: trainer。
- GPU 1: rollout/vLLM server，或双卡 DDP/Accelerate。
- GRPO 初期优先使用较小 `num_generations=4`、`max_completion_length=512`、`max_prompt_length=3072`，稳定后再扩展。

路径规划：

- 代码仓库：`/root/autodl-fs/repos/appworld-agentic-rl/`
- 长期数据：`/root/autodl-fs/datasets/`
- 长期结果：`/root/autodl-fs/outputs/`
- Hugging Face 缓存：`/root/autodl-tmp/hf_home/`
- 高 IO 临时文件：`/root/autodl-tmp/scratch/`

环境规划：

- 当前主环境：`appworld_rl`，Python 3.11，包含 AppWorld、Transformers、TRL、PEFT、bitsandbytes，用于 C0/C1 和后续小规模 dry run。
- 若后续 BFCL/其它 benchmark 与 AppWorld 依赖冲突，再拆出 `agentic_rl` 训练环境，通过 rollout/reward 文件与 `appworld_rl` 交换数据。
- 长任务统一用 `tmux`，日志写到 `/root/autodl-fs/outputs/logs/`。
- 重要 ckpt、rollout、评估报告写到 `/root/autodl-fs/outputs/`，不要只留在 `/root/autodl-tmp/`。

## 3. 推荐目录结构

```text
agentic_rl/
  configs/
    env/
    sft/
    rollout/
    reward/
    grpo/
    eval/
  data/
    raw/
    appworld/
    processed/
    sft/
    rollouts/
    rewards/
    eval/
  reports/
    env_audit.md
    base_model_audit.md
    reward_design.md
    capability_boundary.md
    failure_analysis.md
  scripts/
    00_install_check.py
    01_appworld_audit.py
    02_base_audit.py
    03_build_sft_data.py
    04_train_sft.py
    05_collect_rollouts.py
    06_score_rollouts.py
    07_train_grpo.py
    08_evaluate.py
    09_failure_analysis.py
  src/
    appworld_rl/
      appworld_client.py
      api_schema.py
      prompting.py
      parsing.py
      execution_trace.py
      sft_builder.py
      curriculum.py
      rewards.py
      rollout.py
      metrics.py
      failure_taxonomy.py
      plotting.py
  requirements.txt
  README.md
```

## 4. 课程学习总体设计

课程学习的目标是避免 7B 模型一开始就在长程、多 App、高副作用风险任务上产生大量全 0 rollout，从而导致 SFT 学不到关键模式、GRPO 缺少 reward 方差信号。

课程不只按 easy/medium/hard 划分，而是同时按六个维度调度：

- 任务难度：easy -> medium -> hard。
- App 数量：single-app -> two-app -> multi-app。
- 行为类型：query-only -> state-modification -> high collateral risk。
- API 文档规模：gold APIs -> retrieved APIs -> retrieved + distractor APIs。
- 参数来源：instruction-grounded -> observation-grounded -> mixed grounding。
- 轨迹质量：reference solution -> successful rollout -> high-variance rollout group。

## 5. 课程阶段与晋级闸门

### C0: 环境与参考解验证

目标：确认 AppWorld 可运行、可重置、可评分。

任务范围：

- 10-20 个 easy 任务。
- 只运行 reference solution 和 dummy wrong action。

产物：

- `reports/env_audit.md`
- reference solution 执行日志。
- evaluator 成功/失败样例。
- dummy rollout 样例。

晋级条件：

- AppWorld 安装、`appworld install`、`appworld download data` 成功。
- reference solution 成功率接近 100%。
- evaluator 能检测错误状态和 collateral damage。
- rollout 日志 schema 固定。

### C1: 基座模型能力审计课程

目标：选择主模型，并知道后训练优先补什么能力。

候选：

- `Qwen/Qwen2.5-Coder-7B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `deepseek-ai/deepseek-coder-6.7b-instruct`

任务范围：

- 30-50 个 easy/medium 任务。
- single-app 优先，少量 two-app。
- 每个模型 zero-shot 和 1-shot 各跑一轮。

指标：

- code block 解析率。
- Python 执行率。
- valid API call rate。
- 参数名/类型合法率。
- complete_task rate。
- task success rate。
- failure category 分布。

晋级条件：

- 选出主模型。
- 明确前三大失败类型。
- 形成 `reports/base_model_audit.md`。

### C2: SFT-v1 基础模仿课程

目标：让模型掌握 AppWorld code-as-action 基本格式。

数据：

- 输入：任务 instruction + 必要全局规则 + API 文档摘要。
- 输出：完整 reference solution code。
- 优先 easy single-app 和短程任务。

Kaggle dry run：

- 50-200 条样本。
- `seq_len=2048/3072`。
- LoRA rank 8 或 16。

AutoDL 正式训练：

- 全部 train 中可解析 reference solution 样本。
- `seq_len=4096` 起步。
- LoRA rank 16 或 32。

晋级条件：

- code block 解析率明显高于 Base。
- Python 执行率明显高于 Base。
- valid API call rate 有提升。

### C3: API-aware SFT 课程

目标：从“会写 AppWorld 风格代码”推进到“会选择 API、填对参数、利用返回 ID”。

数据输入包含三类 API：

- Gold APIs: reference solution 实际使用的 API。
- Retrieved APIs: 根据 instruction 检索出的候选 API。
- Distractor APIs: 同 app 或相似功能的干扰 API。

课程调度：

- C3.1: 只给 Gold APIs。
- C3.2: Gold APIs + Retrieved APIs。
- C3.3: Gold APIs + Retrieved APIs + 少量 Distractor APIs。

样本字段：

```json
{
  "task_id": "string",
  "split": "train",
  "instruction": "string",
  "apps": ["calendar"],
  "difficulty": "easy",
  "api_context_type": "gold_plus_retrieved",
  "selected_api_docs": [],
  "used_apis": [],
  "argument_sources": [],
  "reference_solution": "string",
  "messages": []
}
```

晋级条件：

- valid API call rate 提升。
- 参数名和参数类型错误下降。
- 相同任务下无关 API 调用减少。

### C4: Step-wise/Trajectory SFT 课程

目标：训练模型利用 observation，而不是只背完整 solution。

数据来源：

- reference solution 运行日志。
- 成功 rollout。
- 可拆分的 query -> observation -> modification 轨迹。

课程调度：

- C4.1: query-only 轨迹。
- C4.2: query + select entity。
- C4.3: query + modify target object。
- C4.4: observation-grounded 参数，例如 event_id、order_id、message_id。

样本格式：

```json
{
  "task_id": "string",
  "step_index": 2,
  "history": [
    {"role": "user", "content": "task instruction"},
    {"role": "assistant", "content": "previous code"},
    {"role": "tool", "content": "observation"}
  ],
  "target_action": "next code block"
}
```

晋级条件：

- observation-grounded 参数错误下降。
- entity selection error 下降。
- complete_task 时机更稳定。

### C5: Rollout 采样课程

目标：为 GRPO 收集有 reward 方差的 rollout group。

采样课程：

- C5.1: easy single-app，K=4，max_steps=3。
- C5.2: easy/medium single-app，K=4-8，max_steps=5。
- C5.3: medium two-app，K=8，max_steps=5。
- C5.4: high collateral risk 任务，小批量人工审计后加入。

每条 rollout 保存：

```json
{
  "task_id": "string",
  "curriculum_level": "C5.2",
  "prompt": "string",
  "step_history": [],
  "model_outputs": [],
  "parsed_code": [],
  "execution_results": [],
  "api_calls": [],
  "complete_task_called": true,
  "evaluator_result": {},
  "reward": 0.0,
  "failure_category": "ArgumentGroundingError"
}
```

rollout group 筛选：

- 保留 reward 有方差的 group。
- 低比例保留全 1 group，用于防止遗忘。
- 全 0 group 只保留可诊断样本；如果比例过高，回退到 C3/C4。

晋级条件：

- 至少 25%-40% 的 task group 存在 reward 方差。
- 执行全失败 group 显著下降。
- easy 任务上出现稳定正 reward。

### C6: GRPO 课程

目标：用相对优势学习提升状态正确性、参数 grounding 和副作用控制。

训练顺序：

1. `SFT + GRPO final-only`
2. `SFT + GRPO execution-aware`
3. `SFT + GRPO state-aware`

课程调度：

- C6.1: easy single-app + final-only reward，确认 GRPO pipeline 正常。
- C6.2: easy/medium + execution-aware reward，降低执行错误。
- C6.3: medium + state-aware reward，优化状态正确性。
- C6.4: 加入 high collateral risk 任务，重点惩罚副作用。

GRPO 初始建议：

```text
num_generations: 4
max_prompt_length: 3072
max_completion_length: 512
temperature: 0.7
top_p: 0.95
learning_rate: 5e-6 ~ 1e-5
beta/KL: small
LoRA rank: 16 or 32
```

晋级条件：

- dev success rate 高于 SFT-only。
- collateral damage rate 不上升，最好下降。
- reward hacking 样例可控。
- ArgumentGroundingError 和 EntitySelectionError 下降。

## 6. Reward 设计

### Reward-v1: Final-only

```text
success: 1.0
failure: 0.0
```

用途：干净 baseline，判断稀疏 reward 是否足够。

### Reward-v2: Execution-aware

```text
code_block_parse_success: +0.05
python_execution_success: +0.10
valid_api_name: +0.10
valid_argument_name: +0.10
valid_argument_type: +0.10
execution_error: -0.20
invalid_api: -0.20
missing_or_bad_argument: -0.20
```

约束：

- 该 reward 不能超过最终成功奖励。
- 重复无意义 API 调用不累计加分。
- complete_task 不能单独给高分。

### Reward-v3: State-aware

```text
task_success: +1.00
partial_tests_passed: +0.20 ~ +0.80
correct_target_object_modified: +0.40
correct_key_field_modified: +0.30
correct_entity_queried: +0.20
wrong_object_modified: -0.50
collateral_damage: -0.50 ~ -1.00
premature_complete_task: -0.30
missing_complete_task: -0.20
invalid_execution: -0.20
```

总 reward：

```text
R = R_final
  + alpha * R_partial
  + beta  * R_execution
  + gamma * R_argument
  - lambda * R_collateral
  - mu * R_invalid
```

初始权重建议：

```text
alpha = 0.5
beta = 0.2
gamma = 0.4
lambda = 1.0
mu = 0.5
```

## 7. 评估体系

总体指标：

- Task success rate。
- Average reward。
- Executable code rate。
- Valid API call rate。
- Argument name accuracy。
- Argument type accuracy。
- Argument value grounding accuracy。
- complete_task rate。
- Collateral damage rate。
- Average steps。
- Invalid action rate。
- Execution error rate。

分组评估：

- easy / medium / hard。
- single-app / multi-app。
- query-only / state-modification。
- short-horizon / long-horizon。
- low API count / high API count。
- instruction-grounded / observation-grounded。
- low collateral risk / high collateral risk。

失败类型：

- FormatError。
- ExecutionError。
- InvalidAPIError。
- ArgumentNameError。
- ArgumentTypeError。
- ArgumentGroundingError。
- EntitySelectionError。
- IncompleteTask。
- PrematureComplete。
- MissingCompleteTask。
- CollateralDamage。
- LongHorizonFailure。

## 8. 实验矩阵

### 实验 A: 基座模型审计

对比：

- Qwen2.5-Coder-7B-Instruct。
- Qwen2.5-7B-Instruct。
- DeepSeek-Coder-6.7B-Instruct。

输出：

- 选择主模型。
- 训练前失败类型分布。

### 实验 B: SFT 数据形式对比

对比：

- Base。
- SFT-v1。
- SFT-v2 API-aware。
- SFT-v3 step-wise，可选。

输出：

- SFT 主要修复哪些错误。
- API-aware 是否显著降低 API/参数错误。

### 实验 C: 课程学习消融

对比：

- 无课程：混合所有任务训练/采样。
- 难度课程：easy -> medium -> hard。
- 多维课程：难度 + App 数 + 参数来源 + 副作用风险。

输出：

- rollout 全 0 group 比例。
- reward 方差 group 比例。
- dev success rate。
- collateral damage rate。

### 实验 D: Reward 消融

对比：

- SFT-only。
- SFT + GRPO final-only。
- SFT + GRPO execution-aware。
- SFT + GRPO state-aware。

输出：

- state-aware reward 是否真正提升状态正确性。
- execution-aware 是否只提升格式/API 合法率。
- 是否出现 reward hacking。

### 实验 E: 能力边界分析

输出：

- Base/SFT/GRPO 错误迁移趋势。
- 不同任务组的成功率。
- 参数 grounding 和 multi-app 是否仍是核心瓶颈。

## 9. 推荐执行顺序

### 第 1 周: 环境与数据

1. 在 Kaggle 安装 AppWorld。
2. 下载数据并固定 `APPWORLD_ROOT`。
3. 跑通 reference solution。
4. 固定 rollout/eval 日志 schema。

推荐先运行：

```bash
pip install -r requirements.txt -i https://pypi.org/simple
appworld install
appworld download data
python scripts/00_install_check.py --run-verify-tests
python scripts/01_appworld_audit.py --split train --num-tasks 3 --run-verify-tasks
```

如果 Kaggle 中 `appworld` 命令不可用，改用：

```bash
python -m appworld.cli install
python -m appworld.cli download data
python scripts/00_install_check.py
python scripts/01_appworld_audit.py --split train --num-tasks 3
```

### 第 2 周: 基座审计

1. 抽取 30-50 个任务。
2. 跑三类候选模型。
3. 输出 base audit report。
4. 确认主模型和 SFT 优先目标。

### 第 3 周: SFT 数据构造

1. 构造 SFT-v1。
2. 解析 reference solution 中的 used APIs。
3. 构造 API-aware SFT-v2。
4. 初步标注 argument source。

### 第 4 周: SFT 训练

1. Kaggle dry run。
2. AutoDL 正式 QLoRA-SFT。
3. 对比 Base/SFT-v1/SFT-v2。
4. 选择 GRPO 初始化 checkpoint。

### 第 5 周: Rollout 与 Reward

1. 从 easy 任务开始 rollout。
2. 计算 final/execution/state-aware reward。
3. 做 rollout group 方差分析。
4. 筛选 GRPO 训练集。

### 第 6 周: GRPO 训练

1. final-only 小规模 sanity check。
2. execution-aware GRPO。
3. state-aware GRPO。
4. 监控 reward hacking 和 collateral damage。

### 第 7 周: 完整评估

1. Base/SFT/GRPO 统一评估。
2. 分组指标统计。
3. 失败类型归因。
4. 画图和生成报告。

### 第 8 周: README 与项目包装

1. 整理复现实验命令。
2. 完成 README。
3. 完成简历版项目描述。
4. 总结有效结论和失败边界。

## 10. 平台安装建议

### Kaggle

Kaggle 作为备选环境。更推荐优先使用干净的 24GB GPU 租机。若使用 Kaggle，必须使用独立虚拟环境运行 AppWorld。Kaggle/Colab 预装环境里很多包依赖 Pydantic v2，而 AppWorld 0.1.x 依赖 Pydantic v1；如果直接在全局环境安装，会出现大量依赖冲突警告。

```bash
cd /kaggle/working/<your-repo-dir>
PROJECT_DIR="$(pwd)"
python -m pip install --user virtualenv
python -m virtualenv /kaggle/working/appworld_venv
/kaggle/working/appworld_venv/bin/python -m pip install --upgrade pip setuptools wheel
/kaggle/working/appworld_venv/bin/python -m pip install wrapt
/kaggle/working/appworld_venv/bin/python -m pip install -r "$PROJECT_DIR/requirements.txt"
/kaggle/working/appworld_venv/bin/appworld install
/kaggle/working/appworld_venv/bin/appworld download data
```

如 `appworld` 命令不可用，使用：

```bash
/kaggle/working/appworld_venv/bin/python -m appworld.cli install
/kaggle/working/appworld_venv/bin/python -m appworld.cli download data
```

### AutoDL

建议先安装匹配 CUDA 的 PyTorch，再安装项目依赖。

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt -i https://pypi.org/simple
appworld install
appworld download data
```

如使用 vLLM 加速 GRPO rollout：

```bash
pip install vllm
trl vllm-serve --model Qwen/Qwen2.5-Coder-7B-Instruct
```

## 11. 风险与回退策略

- 如果 Base 几乎无法产生可执行代码，先加强 SFT-v1 格式课程。
- 如果 SFT 后 API 合法率仍低，扩大 API-aware SFT，并减少 API 文档干扰。
- 如果 observation-grounded 参数错误高，优先构造 C4 step-wise 样本。
- 如果 rollout group 大量全 0，降低任务难度，缩短 max_steps，回到 easy single-app。
- 如果 GRPO 后 collateral damage 上升，提高 state-aware penalty，减少 high-risk 任务比例。
- 如果 final-only GRPO 不提升，不直接否定 GRPO，先检查 reward 方差和 rollout 质量。

## 12. 最终交付物

- AppWorld 环境与数据审计报告。
- 基座模型能力审计报告。
- API-aware SFT 数据构造说明。
- SFT-v1/v2/v3 数据样例。
- QLoRA-SFT checkpoint。
- rollout 数据集。
- reward 设计文档。
- QLoRA-GRPO checkpoint。
- Base/SFT/GRPO 对比评估报告。
- 课程学习消融报告。
- 能力边界分析报告。
- 失败类型分析报告。
- README 与简历项目描述。

## 13. 参考资料

- AppWorld GitHub: https://github.com/StonyBrookNLP/appworld
- AppWorld website: https://appworld.dev/
- TRL GRPOTrainer documentation: https://huggingface.co/docs/trl/grpo_trainer
