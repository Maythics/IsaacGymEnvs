# 固定 Base 的 Gravity-in-Palm 课程

`Shadowhand18Gravity` 与 `WujiHandGravity` 不旋转 hand root/base，也不按目标重力
调整 object 的 spawn pose。它们只对物理 object 施加补偿力，使 object 的实际加速度
从模拟器重力平滑切换到指定重力。

`WujiHandGravity` 使用与 `WujiHandFixedTilt` 相同的 fixed-wrist compatibility
asset：22 维 checkpoint 接口仍存在，但 wrist compatibility DOF 不连接真实 palm，因而
课程中的 `gravityInPalm` 不会被 wrist motion 额外改变。

## 坐标与符号

`task.env.gravityInPalm=[x,y,z]` 是 **物理向下加速度** 的单位向量，坐标系是任务启动后
读取的初始 palm rigid-body frame。不是 world frame，也不是旧 tilt manifest 中的
“up”向量。

```text
g_target_world = R_world_from_initial_palm * (9.81 * gravityInPalm)
force_on_object = mass * (g_effective_world - sim.gravity)
```

普通任务的模拟器重力固定是 `sim.gravity=[0,0,-9.81]`，但它在两种 palm frame 中并不相同：

| Task | 标准 asset 的 native gravity-in-palm |
| --- | --- |
| `Shadowhand18Gravity` | `[0,+1,0]` |
| `WujiHandGravity` | `[-1,0,0]` |

这些数值由 task 在运行时从 resolved palm pose 再计算并打印；asset、初始 wrist pose 或
`palmBodyName` 改动后，应以该运行时值为准。`gravityInPalm: null` 使用该 native 值，
因此与普通 task 的 object 重力一致。

## Reset 时序

所有 target 都采用同一个时序：

```text
0.0s <= t < 0.2s : g_effective_world = sim.gravity（默认 world -Z）
0.2s <= t < 0.4s : 在 sim.gravity 与 g_target_world 之间线性插值
t >= 0.4s         : g_effective_world = g_target_world
```

这里的“沿 z 轴”专指世界系的默认 `sim.gravity`，不能写成统一的 palm `[0,0,-1]`。

## 查看与课程命令

先在 Isaac Gym 环境中检查普通条件和一个横向条件：

```bash
cd /home/srtp/research_manip/IsaacGymEnvs/isaacgymenvs
python train.py task=Shadowhand18Gravity checkpoint=/path/to/Shadowhand18.pth \
  num_envs=64 test=True headless=False task.env.gravityInPalm='[0,1,0]'

python train.py task=WujiHandGravity checkpoint=/path/to/WujiHand.pth \
  num_envs=64 test=True headless=False task.env.gravityInPalm='[0,0,-1]'
```

## 四 GPU 连续课程（ShadowHand18）

默认 seed 已写入 `curricula/gravity_42.yaml`，并且是相对该 YAML 的路径：

```text
../runs/Shadowhand18_25-18-08-31_0/nn/Shadowhand18.pth
```

课程使用 GPU `0,1,2,3` 同时启动四个 trainer。每个 target 从已成功的、球面距离最近的
checkpoint warm start；canonical 42 点的最小相邻间隔为约 `22.062°`，因此只有距离不超过两倍
间隔、即约 `44.124°` 的 parent checkpoint 可以复用。没有合格 parent 的 target 会保持 pending，
不会跨越球面远距离加载 checkpoint。

先检查命令、GPU 分配与 seed 路径，不启动训练：

```bash
cd /home/srtp/research_manip/IsaacGymEnvs/isaacgymenvs
python scripts/run_gravity_curriculum.py --task Shadowhand18Gravity --dry-run
```

正式启动（四张 GPU）：

```bash
cd /home/srtp/research_manip/IsaacGymEnvs/isaacgymenvs
python scripts/run_gravity_curriculum.py --task Shadowhand18Gravity
```

状态会持久化到 `isaacgymenvs/curriculum_runs/gravity_42_Shadowhand18Gravity/state.json`，路径和
既有 state JSON 语义不变。按 `Ctrl-C` 时 launcher 会将它自己启动的 worker 停止，并把 running
cell 记为 pending；重新运行相同命令时，该 target 会先检索已有训练输出，再从其中恢复。

若第一轮有未达到 `score_to_win`、但已经接近成功的 cell，使用：

```bash
python scripts/run_gravity_curriculum.py --task Shadowhand18Gravity --retry all
# 也支持：python scripts/run_gravity_curriculum.py --task Shadowhand18Gravity --retry --all
```

`--retry all` 只重新排队 **未成功** 的 target，不会重训已通过 score gate、已经作为 parent 的
target。对于每一个已尝试 target，launcher 会扫描
`runs/gravity_*_<target-id>_a*/nn/`。rl_games 约定的最佳模型是与 experiment 同名、且不带
`last` 的文件：例如 `gravity_10_p180_t120_a01/nn/gravity_10_p180_t120_a01.pth`；launcher
会直接优先选它，绝不以 `*_last.pth` 覆盖它。多个历史 attempt 都有该 best 文件时，选择最近
写入的一个。`last_mean_rewards` 只用于本次训练结束后的 `score_to_win` gate，不用于恢复模型
选择。没有找到该 target 的有效 best checkpoint 时，才回退到原有的相邻成功 parent 选择逻辑。

调试时可限制本次最多启动的 cell 数，例如：

```bash
python scripts/run_gravity_curriculum.py --task Shadowhand18Gravity --max-targets 4
```

每个成功 trainer 必须以退出码 0 结束，并在对应 `runs/<experiment>/nn/` 目录产生 `.pth`
checkpoint；launcher 会读取 checkpoint 内的 `last_mean_rewards`，只有严格大于
`score_to_win: 2500` 才会晋升为 parent。每个 worker 同时沿用 remote curriculum 的
`save_best_after: 1`、`timeout_seconds: 10800` 和按 `nvidia-smi` 空闲显存选择的
`resource_profiles`；空闲显存低于 4096 MiB 时不会派发 job。大规模训练前仍应先在 viewer
检查 reset pose 未变、前 0.2 秒可抓取、以及 0.2--0.4 秒没有接触跳变。

## 范围

本模块不修改 `Shadowhand18Tilted`、`WujiHandFixedTilt`、旧 tilt curriculum 或任何
ManiSkill rollout/H5/diffusion 代码。固定 base 后，后续数据收集必须显式记录 object 的
有效重力，不能再只通过 palm pose 和 world gravity 推导；该数据契约不属于本次改动。
