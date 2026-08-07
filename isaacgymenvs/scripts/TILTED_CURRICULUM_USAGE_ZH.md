# 倾斜手腕任务与自动课程：中文使用手册

本文对应两个保持旧 checkpoint 接口不变的任务：

- `Shadowhand18Tilted`：209 维 observation、18 维 action；只能加载兼容的 `Shadowhand18` / `Shadowhand18Tilted` checkpoint。
- `WujiHandFixedTilt`：207 维 observation、22 维 action；可加载普通 `WujiHand` checkpoint。

请先进入训练脚本所在目录：

```bash
cd /home/srtp/research_manip/IsaacGymEnvs/isaacgymenvs
```

## 1. 先在 viewer 中逐个检查 offset

不要直接用大规模 curriculum 去猜 offset。先使用 `num_envs=64`、`test=True`、`headless=False` 在 viewer 中观察：物体是否一开始就在手心附近、是否被手指顶飞、以及在重力恢复后是否仍可保持。`objectPalmOffset` 是**手掌局部坐标系**的平移。

当前默认的重力辅助为：保持 `0.2 s` 完全抵消物体重力，再用 `0.1 s` 线性恢复。它不会暂停 viewer，也不会冻结 policy；这段时间内 policy 和接触力仍然正常工作，只是物体不因重力下落。为了每次检查的物理条件明确，下面的命令都显式写出这两个参数。

### WujiHandFixedTilt：从已知的 0 度 checkpoint 开始

下列命令是 30 度、`+Y` 倾斜轴的已调 offset。`baseTiltAxis` 不要省略；若省略，虽然当前默认也是 `+Y`，但不利于复现实验。

```bash
python train.py task=WujiHandFixedTilt \
  num_envs=64 \
  'train.params.config.minibatch_size=64' \
  max_iterations=200000 \
  task.env.objectType=block \
  checkpoint=/home/srtp/research_manip/IsaacGymEnvs/isaacgymenvs/runs/WujiHand_29-10-01-07/nn/WujiHand.pth \
  task.env.baseTiltAngleDeg=30 \
  task.env.baseTiltAxis='[0.0,1.0,0.0]' \
  task.env.objectPalmOffset='[0.0,0.05,0.0]' \
  task.env.objectGravityCompensationSeconds=0.2 \
  task.env.objectGravityRampSeconds=0.1 \
  test=True headless=False
```

要检查任意角度时，只替换 `baseTiltAngleDeg`、`baseTiltAxis`、`objectPalmOffset`。如果得到更好的 offset，请先写回 `curricula/wujihand_fixed_tilt_42.yaml` 的对应已验证目标；不要用连续插值替换手工确认的值。

### Shadowhand18Tilted：使用兼容的 Shadowhand18 checkpoint

先把 `SH18_CKPT` 改成实际存在的、18-action Shadowhand18 checkpoint。不能把普通 20-action `ShadowHand` checkpoint 用在这里。

```bash
export SH18_CKPT=/absolute/path/to/compatible/Shadowhand18.pth

python train.py task=Shadowhand18Tilted \
  num_envs=64 \
  'train.params.config.minibatch_size=64' \
  max_iterations=200000 \
  task.env.objectType=block \
  checkpoint="$SH18_CKPT" \
  task.env.baseTiltAngleDeg=30 \
  task.env.baseTiltAxis='[0.0,1.0,0.0]' \
  task.env.objectPalmOffset='[0.08,0.0,-0.01]' \
  task.env.objectGravityCompensationSeconds=0.2 \
  task.env.objectGravityRampSeconds=0.1 \
  test=True headless=False
```

角度约定如下。第一个是倾斜量 `theta`，第二个是方位角 `phi`；手腕轴固定由方位角决定：

```text
baseTiltAngleDeg = theta
baseTiltAxis     = [-sin(phi), cos(phi), 0]
```

例如 `phi=90°` 的轴是 `[-1,0,0]`，`phi=180°` 的轴是 `[0,-1,0]`。这也是下方所有命令使用的轴。

## 2. Shadowhand18Tilted 的全部 108 条人工检查命令

以下 108 条是 block 阶段的全部真实角度训练/认证案例：前 42 条为人工验证方向，后 66 条为 15° 球面覆盖所增加的均匀 maximin 方向。它们不是 bridge；每一条都会单独训练、单独评估。新增方向的 offset 直接继承最近的人工验证方向，不做插值。

每行最后一个数字是 `baseYawDeg`。它是按这份手册的前一条 viewer 姿态计算的连续 world-Z yaw；因此请顺序执行。它不改变重力在掌心内的方向或 offset，只消除坐标表示额外引入的世界姿态跳变。尤其是第 006 条到第 007 条必须使用 `-90`，不能省略第四个参数。

先执行一次这个 shell 函数；之后本节每一行都是可直接复制执行的检查命令。将 `SH18_CKPT` 替换为你实际可用的 checkpoint。要连续观察下一条前，请先关闭前一个 viewer 进程。

```bash
export SH18_CKPT=/absolute/path/to/compatible/Shadowhand18.pth

run_shadow18_case() {
  python train.py task=Shadowhand18Tilted \
    num_envs=64 \
    'train.params.config.minibatch_size=64' \
    max_iterations=200000 \
    task.env.objectType=block \
    checkpoint="$SH18_CKPT" \
    task.env.baseTiltAngleDeg="$1" \
    task.env.baseTiltAxis="$2" \
    task.env.objectPalmOffset="$3" \
    task.env.baseYawDeg="$4" \
    task.env.objectGravityCompensationSeconds=0.2 \
    task.env.objectGravityRampSeconds=0.1 \
    test=True headless=False
}
```

```bash
# 001 p000_t030
run_shadow18_case 30 '[0,1,0]' '[0.08,0,-0.01]' 0
# 002 p000_t060
run_shadow18_case 60 '[0,1,0]' '[0.08,0.06,0]' 0
# 003 p000_t090
run_shadow18_case 90 '[0,1,0]' '[0.08,0.06,0]' 0
# 004 p000_t120
run_shadow18_case 120 '[0,1,0]' '[0.08,0.09,0.01]' 0
# 005 p000_t150
run_shadow18_case 150 '[0,1,0]' '[0.08,0.09,0.01]' 0
# 006 south_pole
run_shadow18_case 180 '[0,1,0]' '[0.08,0.09,0.01]' 0
# 007 p045_t150
run_shadow18_case 150 '[-0.70710678,0.70710678,0]' '[0.08,0.1,0.01]' -90
# 008 p045_t120
run_shadow18_case 120 '[-0.70710678,0.70710678,0]' '[0.08,0.08,0.01]' -90
# 009 p045_t090
run_shadow18_case 90 '[-0.70710678,0.70710678,0]' '[0.08,0.08,0.01]' -90
# 010 p045_t060
run_shadow18_case 60 '[-0.70710678,0.70710678,0]' '[0.1,0.06,0.01]' -90
# 011 p045_t030
run_shadow18_case 30 '[-0.70710678,0.70710678,0]' '[0.08,0.09,0.01]' -90
# 012 p090_t030
run_shadow18_case 30 '[-1,0,0]' '[0.08,0.09,0.01]' -95.532203
# 013 p090_t060
run_shadow18_case 60 '[-1,0,0]' '[0.08,0.09,0.01]' -95.532203
# 014 p090_t090
run_shadow18_case 90 '[-1,0,0]' '[0.08,0.09,0.01]' -95.532203
# 015 p090_t120
run_shadow18_case 120 '[-1,0,0]' '[0.08,0.09,0.01]' -95.532203
# 016 p090_t150
run_shadow18_case 150 '[-1,0,0]' '[0.08,0.09,0.01]' -95.532203
# 017 p135_t150
run_shadow18_case 150 '[-0.70710678,-0.70710678,0]' '[0.03,0.08,0.07]' -180
# 018 p135_t120
run_shadow18_case 120 '[-0.70710678,-0.70710678,0]' '[0.03,0.08,0.07]' -180
# 019 p135_t090
run_shadow18_case 90 '[-0.70710678,-0.70710678,0]' '[0.03,0.08,0.07]' -180
# 020 p135_t060
run_shadow18_case 60 '[-0.70710678,-0.70710678,0]' '[0,0.04,0.04]' -180
# 021 p135_t030
run_shadow18_case 30 '[-0.70710678,-0.70710678,0]' '[-0.04,0,0]' -180
# 022 p180_t030
run_shadow18_case 30 '[0,-1,0]' '[0.04,0.09,0.01]' 174.4678
# 023 p180_t060
run_shadow18_case 60 '[0,-1,0]' '[0.04,0.09,0.01]' 174.4678
# 024 p180_t090
run_shadow18_case 90 '[0,-1,0]' '[0,0.09,0.01]' 174.4678
# 025 p180_t120
run_shadow18_case 120 '[0,-1,0]' '[0,0.09,0.01]' 174.4678
# 026 p180_t150
run_shadow18_case 150 '[0,-1,0]' '[0.04,0.09,0.01]' 174.4678
# 027 p225_t150
run_shadow18_case 150 '[0.70710678,-0.70710678,0]' '[0.04,0.09,0]' 90.000003
# 028 p225_t120
run_shadow18_case 120 '[0.70710678,-0.70710678,0]' '[0.04,0.09,0]' 90.000003
# 029 p225_t090
run_shadow18_case 90 '[0.70710678,-0.70710678,0]' '[0.04,0.08,0]' 90.000003
# 030 p225_t060
run_shadow18_case 60 '[0.70710678,-0.70710678,0]' '[0.04,0.05,0]' 90.000003
# 031 p225_t030
run_shadow18_case 30 '[0.70710678,-0.70710678,0]' '[0.04,0.05,0]' 90.000003
# 032 p270_t030
run_shadow18_case 30 '[1,0,0]' '[0.04,0.05,0]' 84.4678
# 033 p270_t060
run_shadow18_case 60 '[1,0,0]' '[0.04,0.08,0]' 84.4678
# 034 p270_t090
run_shadow18_case 90 '[1,0,0]' '[0.04,0.08,0]' 84.4678
# 035 p270_t120
run_shadow18_case 120 '[1,0,0]' '[0.04,0.08,0]' 84.4678
# 036 p270_t150
run_shadow18_case 150 '[1,0,0]' '[0.06,0.08,0]' 84.4678
# 037 p315_t150
run_shadow18_case 150 '[0.70710678,0.70710678,0]' '[0.06,0.08,0]' 3.0707861e-06
# 038 p315_t120
run_shadow18_case 120 '[0.70710678,0.70710678,0]' '[0.06,0.08,0]' 3.0707861e-06
# 039 p315_t090
run_shadow18_case 90 '[0.70710678,0.70710678,0]' '[0.06,0.08,0]' 3.0707861e-06
# 040 p315_t060
run_shadow18_case 60 '[0.70710678,0.70710678,0]' '[0.06,0.08,0]' 3.0707861e-06
# 041 p315_t030
run_shadow18_case 30 '[0.70710678,0.70710678,0]' '[0.06,0.08,0]' 3.0707861e-06
# 042 north_pole
run_shadow18_case 0 '[0,1,0]' '[0,0,0]' 3.0707861e-06
# 043 dense_001
run_shadow18_case 76.853011 '[0.92372775,-0.38304965,0]' '[0.04,0.08,0]' 3.0707861e-06
# 044 dense_002
run_shadow18_case 76.74708 '[0.92496094,0.38006217,0]' '[0.04,0.08,0]' -34.090705
# 045 dense_003
run_shadow18_case 102.98818 '[0.3812911,0.92445503,0]' '[0.08,0.06,0]' -79.226201
# 046 dense_004
run_shadow18_case 76.958897 '[0.38456986,-0.92309589,0]' '[0.04,0.08,0]' 55.612037
# 047 dense_005
run_shadow18_case 76.641102 '[0.38755129,0.92184814,0]' '[0.06,0.08,0]' -21.731863
# 048 dense_006
run_shadow18_case 102.88236 '[-0.38182169,0.92423601,0]' '[0.08,0.06,0]' -66.776726
# 049 dense_007
run_shadow18_case 103.09404 '[0.92236238,0.38632582,0]' '[0.06,0.08,0]' 48.158016
# 050 dense_008
run_shadow18_case 76.535079 '[-0.37554543,0.92680399,0]' '[0.08,0.06,0]' -40.797287
# 051 dense_009
run_shadow18_case 104.7884 '[-0.38341931,-0.92357438,0]' '[0.03,0.08,0.07]' -179.51991
# 052 dense_010
run_shadow18_case 104.68176 '[0.37969196,-0.92511298,0]' '[0,0.09,0.01]' 123.63256
# 053 dense_011
run_shadow18_case 76.429008 '[-0.91994659,0.39204371,0]' '[0.08,0.08,0.01]' -98.179178
# 054 dense_012
run_shadow18_case 103.19995 '[0.92630407,-0.37677681,0]' '[0.04,0.08,0]' 37.023109
# 055 dense_013
run_shadow18_case 77.064737 '[-0.37853889,-0.92558539,0]' '[0,0.09,0.01]' 127.40438
# 056 dense_014
run_shadow18_case 102.77659 '[-0.92258398,0.38579631,0]' '[0.08,0.08,0.01]' -142.30968
# 057 dense_015
run_shadow18_case 104.89509 '[-0.92324972,-0.3842004,0]' '[0.03,0.08,0.07]' 161.0073
# 058 dense_016
run_shadow18_case 76.322889 '[-0.92862497,-0.37101976,0]' '[0.08,0.09,0.01]' 161.83183
# 059 dense_017
run_shadow18_case 47.324138 '[-0.92382606,-0.38281251,0]' '[0,0.04,0.04]' 161.45814
# 060 dense_018
run_shadow18_case 47.183701 '[-0.38480682,-0.92299714,0]' '[-0.04,0,0]' 147.90136
# 061 dense_019
run_shadow18_case 132.4658 '[-0.92350773,-0.38357982,0]' '[0.03,0.08,0.07]' -167.48244
# 062 dense_020
run_shadow18_case 132.60576 '[-0.92517895,0.37953116,0]' '[0.08,0.09,0.01]' -91.437181
# 063 dense_021
run_shadow18_case 47.464258 '[-0.92486335,0.3802996,0]' '[0.08,0.09,0.01]' -91.389544
# 064 dense_022
run_shadow18_case 132.74604 '[-0.38808039,0.92162553,0]' '[0.08,0.08,0.01]' -46.459025
# 065 dense_023
run_shadow18_case 132.32615 '[-0.38403993,-0.92331649,0]' '[0.03,0.08,0.07]' 62.478432
# 066 dense_024
run_shadow18_case 47.042945 '[0.37830128,-0.92568253,0]' '[0.04,0.09,0.01]' 18.019396
# 067 dense_025
run_shadow18_case 132.88664 '[0.37501337,0.9270194,0]' '[0.08,0.09,0.01]' -117.49046
# 068 dense_026
run_shadow18_case 47.604065 '[-0.38731463,0.9219476,0]' '[0.1,0.06,0.01]' -162.57814
# 069 dense_027
run_shadow18_case 132.18681 '[0.37907013,-0.92536795,0]' '[0,0.09,0.01]' 75.800132
# 070 dense_028
run_shadow18_case 46.901865 '[0.92110896,-0.38930487,0]' '[0.04,0.05,0]' 31.49827
# 071 dense_029
run_shadow18_case 133.02755 '[0.9197214,0.39257171,0]' '[0.06,0.08,0]' -14.486565
# 072 dense_030
run_shadow18_case 47.74356 '[0.37578334,0.92670755,0]' '[0.08,0.06,0]' -59.731414
# 073 dense_031
run_shadow18_case 132.04777 '[0.92143205,-0.38853955,0]' '[0.04,0.09,0]' 30.774132
# 074 dense_032
run_shadow18_case 46.760461 '[0.92751697,0.37378104,0]' '[0.04,0.05,0]' -13.370542
# 075 dense_033
run_shadow18_case 163.83213 '[-0.3779448,0.92582813,0]' '[0.08,0.09,0.01]' -157.46128
# 076 dense_034
run_shadow18_case 15.981634 '[-0.37465638,-0.92716374,0]' '[0,0,0]' 68.404099
# 077 dense_035
run_shadow18_case 15.960811 '[-0.35003015,0.93673843,0]' '[0,0,0]' 69.949761
# 078 dense_036
run_shadow18_case 164.1857 '[0.33936885,-0.94065338,0]' '[0.08,0.09,0.01]' 171.73009
# 079 dense_037
run_shadow18_case 104.96032 '[0.09027649,-0.99591674,0]' '[0,0.09,0.01]' -161.755
# 080 dense_038
run_shadow18_case 164.2278 '[-0.90738209,-0.42030673,0]' '[0.08,0.09,0.01]' -32.149421
# 081 dense_039
run_shadow18_case 16.002431 '[0.90255006,0.43058494,0]' '[0.06,0.08,0]' -108.26048
# 082 dense_040
run_shadow18_case 75.116766 '[-0.64018136,-0.76822382,0]' '[0.03,0.08,0.07]' -102.52829
# 083 dense_041
run_shadow18_case 75.010027 '[0.088110293,-0.99611072,0]' '[0.04,0.09,0.01]' -135.2422
# 084 dense_042
run_shadow18_case 105.0671 '[-0.63850906,-0.76961431,0]' '[0.03,0.08,0.07]' -90.473703
# 085 dense_043
run_shadow18_case 75.223452 '[-0.99566905,-0.092968549,0]' '[0.08,0.09,0.01]' -45.364343
# 086 dense_044
run_shadow18_case 104.9366 '[0.084574244,0.99641718,0]' '[0.08,0.06,0]' 55.019595
# 087 dense_045
run_shadow18_case 74.9863 '[0.086741115,0.99623089,0]' '[0.08,0.06,0]' 55.144119
# 088 dense_046
run_shadow18_case 104.85359 '[0.76649032,-0.64225586,0]' '[0.04,0.08,0]' 179.81
# 089 dense_047
run_shadow18_case 75.093051 '[0.76420594,0.64497231,0]' '[0.06,0.08,0]' 99.732587
# 090 dense_048
run_shadow18_case 105.17393 '[-0.9954645,-0.095133773,0]' '[0.08,0.09,0.01]' -46.44754
# 091 dense_049
run_shadow18_case 75.330086 '[-0.77133766,0.63642613,0]' '[0.08,0.08,0.01]' -1.2467095
# 092 dense_050
run_shadow18_case 104.82988 '[-0.64290398,0.76594678,0]' '[0.08,0.08,0.01]' 9.2338975
# 093 dense_051
run_shadow18_case 104.22025 '[0.9931523,0.11682687,0]' '[0.04,0.08,0]' -177.60612
# 094 dense_052
run_shadow18_case 75.726548 '[0.99356069,-0.11330114,0]' '[0.04,0.08,0]' -164.39782
# 095 dense_053
run_shadow18_case 74.903235 '[0.76509169,-0.64392135,0]' '[0.04,0.05,0]' -139.56668
# 096 dense_054
run_shadow18_case 105.04336 '[0.7628014,0.64663283,0]' '[0.06,0.08,0]' 140.10697
# 097 dense_055
run_shadow18_case 90.552913 '[-0.87976601,-0.47540694,0]' '[0.03,0.08,0.07]' -133.3653
# 098 dense_056
run_shadow18_case 164.59021 '[0.92397339,0.38245675,0]' '[0.08,0.09,0.01]' -119.74463
# 099 dense_057
run_shadow18_case 61.97544 '[0.94744935,0.31990582,0]' '[0.04,0.08,0]' -113.49452
# 100 dense_058
run_shadow18_case 62.092208 '[0.89723511,-0.44155311,0]' '[0.04,0.05,0]' -90.545442
# 101 dense_059
run_shadow18_case 117.84946 '[0.89566226,0.44473489,0]' '[0.06,0.08,0]' -143.12221
# 102 dense_060
run_shadow18_case 62.20885 '[0.32452393,-0.94587749,0]' '[0.04,0.09,0.01]' -45.577434
# 103 dense_061
run_shadow18_case 117.73288 '[0.32116469,0.94702336,0]' '[0.08,0.09,0.01]' 172.28574
# 104 dense_062
run_shadow18_case 91.274936 '[-0.26952293,-0.96299397,0]' '[0,0.09,0.01]' 157.18965
# 105 dense_063
run_shadow18_case 91.17178 '[0.48823085,-0.87271452,0]' '[0.04,0.08,0]' 111.31954
# 106 dense_064
run_shadow18_case 62.325367 '[-0.4371709,-0.89937846,0]' '[0,0.04,0.04]' 152.37554
# 107 dense_065
run_shadow18_case 117.61642 '[-0.44036031,0.89782114,0]' '[0.08,0.08,0.01]' -79.809428
# 108 dense_066
run_shadow18_case 117.50009 '[-0.94544536,0.32578071,0]' '[0.08,0.09,0.01]' -146.29347
```

如果其中某一条的实际 viewer 表现不合理，应先手动调该条 offset，再把确认后的 offset 写到 manifest。新增方向当前继承的是 `offset_source` 所指的已验证条目；只有你明确确认后才应改成自己的独立 offset。

### 自动训练为何会额外设置 `baseYawDeg`

上面的人工检查命令使用的是一条固定的、相邻 viewer 姿态连续的 yaw 路径；自动 launcher 则会根据**实际选中的 parent checkpoint**重新精确计算 `baseYawDeg`，并将所选数值写进 `state.json`、训练日志的 `COMMAND:` 行和 evaluator 配置。因此手册中的 yaw 用于连续可视化和调 offset；真正训练时以 launcher 记录的 `COMMAND:` 为准。

这是一个只绕**世界 Z 轴**的旋转：它不改变世界重力在手掌局部坐标中的方向，不改变 `objectPalmOffset`，且对水平地面上的物理是 yaw 对称的；但可以消除旧世界系 MLP 输入中任意的 roll/gauge 跳变。例如历史的 `p000_t180 → p045_t150`：

```text
历史 baseYawDeg=0：手根 SO(3) 距离约 93.84°，而重力方向只差 30°。
自动连续性 yaw：child baseYawDeg=-90°，手根 SO(3) 距离变为 30°。
```

因此不要把 launcher 日志中的 `yaw=...` 当成新的 gravity direction 或新的 offset。它只是让绝对世界坐标 observation 与 parent checkpoint 的输入分布连续；训练目标的 gravity-in-palm 始终是 manifest 指定的 108 个方向。

### 2.1 连续 parent 前沿调度（30° 软目标）

`training.max_parent_transition_deg: 30.0` 是**软连续性目标**，不依赖 YAML 行顺序。每当一个 worker 空闲时，launcher 会把当前 stage 的**全部 pending 目标**都与所有已成功/显式可信 parent 比较，优先选择经过 world-Z yaw 优化后 hand SO(3) 距离不超过 30° 的目标。这样新的 checkpoint 会尽量接在接近的 checkpoint 上，而不是按 `p000 → p045 → ...` 的固定列表突然跳转。

它不是硬门槛：如果某个 `x°` 方向 timeout/failed，`x+15°` 没有近邻 parent 时，空闲 GPU 会立刻从全局最近的成功 checkpoint 启动，而不会等待或卡住整个课程；日志会显示 `fallback beyond 30.00 deg limit`。因此四张 GPU 会持续有可调度目标，fallback 只是“当前成功 checkpoint 覆盖尚不够近”的显式标记，不是 yaw 失效。

`allow_unscored_discovered_parents: false` 还会拒绝自动发现的低 reward 或无法从文件名确认 reward 的历史 checkpoint；例如 `_rew_89` 不会再作为 warm start。你确认过的历史 checkpoint 应保留在 `existing_start_checkpoints`，它们按 manifest 的 `recorded_reward > 2500` 使用。

### 2.2 检查已经收集到的 checkpoint（推荐）

不要根据 `runs/` 目录的时间顺序猜哪个模型可用。每次 curriculum 的唯一记录是 `--state-dir/state.json`；其中保存了该目标实际采用的 parent、实际 `baseYawDeg`、reward、尝试次数、训练日志和输出 checkpoint 路径。日志第一行的 `COMMAND:` 是当时训练的完整命令，因此即使后来改过 YAML，仍能复现这个 checkpoint 的原始物理参数。

以目录名 `sh18tilt_22_p180_t030_a01` 为例：

- `sh18tilt`：ShadowHand18 tilted 课程（Wuji 则为 `wujitilt`）。
- `22`：该目标在运行时 108 个 target 中的编号（从 1 开始），不是 reward 或角度。
- `p180_t030`：目标 ID；`p180` 是方位角 `phi=180°`，`t030` 是倾角 `theta=30°`。
- `a01`：第 1 次尝试；同一 target timeout 或失败后会成为 `a02`、`a03`。应以 `state.json` 中该 target 当前的 `run_name` 和 `output_checkpoint` 为准，不能只按 `a` 最大或目录最新来选。

`dense_001` 至 `dense_066` 不是 bridge，也不是临时模型；它们是运行时在 42 条人工验证方向外按 maximin 球面覆盖补出的 66 个真实训练 target。其倾角、轴、继承的 offset 与实际 yaw 均由 `state.json` / 对应日志确定。

使用下面的工具可以直接列出已晋级的 checkpoint，并从训练日志生成**精确的非 headless viewer 命令**。生成的命令会使用该 checkpoint 的实际 axis、offset、`baseYawDeg`、重力补偿和 checkpoint，不需要再手抄 108 条参数。

```bash
# 默认先列出已达到当前 score_to_win、可作为 parent 的 checkpoint，
# 再列出所有 timeout 但已训练过的 target（用于诊断 offset）。
python scripts/inspect_tilted_curriculum_checkpoints.py \
  --manifest curricula/remote_shadowhand18_tilted.yaml \
  --state-dir /tmp/shadowhand18_tilt_continuous_v1 \
  --format table

# 先输出所有已晋级 checkpoint，再输出 timeout 的最佳已保存 checkpoint；
# 一次只运行一条，关闭 viewer 后再运行下一条。
python scripts/inspect_tilted_curriculum_checkpoints.py \
  --manifest curricula/remote_shadowhand18_tilted.yaml \
  --state-dir /tmp/shadowhand18_tilt_continuous_v1 \
  --format commands > /tmp/sh18_view_succeeded.sh

# 只检查一个 target，例如 dense_045 或 p180_t030。
python scripts/inspect_tilted_curriculum_checkpoints.py \
  --manifest curricula/remote_shadowhand18_tilted.yaml \
  --state-dir /tmp/shadowhand18_tilt_continuous_v1 \
  --target dense_045
```

`--format commands` 的输出本身是 shell 命令，默认用 `num_envs=64`、`test=True`、`headless=False`。默认分两段：先是 `PROMOTED / SUCCEEDED`，随后是 `TIMED OUT`。timeout target 若有 `a01/a02/...` 多次尝试，工具会扫描这些 run 目录，读取已保存 checkpoint 的 `last_mean_rewards`，选择其中 reward 最高的一份；这正是应拿来判断「这个 offset 是否仍然可能训好」的 checkpoint。可先用编辑器打开 `/tmp/sh18_view_succeeded.sh`，复制其中一条运行；或直接 `bash /tmp/sh18_view_succeeded.sh`，每关闭一个 viewer 后会继续下一条。若只想看成功的，添加 `--include succeeded`；若还要检查异常退出但留下 checkpoint 的记录，添加 `--include all-output`。timeout/failed 模型不保证超过当前 `score_to_win`，只适合作诊断。

对于本地 ShadowHand manifest，把 `--manifest` 换成 `curricula/shadowhand18_tilt_42.yaml`；Wuji 则使用 `curricula/wujihand_fixed_tilt_42.yaml` 和对应的 state-dir。若旧日志被手工删除，工具会改用 state 和当前 manifest 生成回退命令；因此要严格复现实验，请保留 `<state-dir>/logs/`。

## 3. 自动课程中的“成功”到底是什么

当前两个 curriculum 的默认 `training.promotion_mode: reward_only`：只要输出 checkpoint 内的 `last_mean_rewards` **严格大于** `score_to_win: 2500`，该方向立即标记为 `succeeded`、释放 worker，并作为后续最近方向的 parent。恰好等于 `2500` 不晋级。自动课程不会再同步运行物理 evaluator，因此 evaluator 的环境/API 异常不会停止或拖慢后续训练。

`evaluate_tilted_policy.py` 仍可作为**可选诊断**使用。它的一个 episode 计为 **retained physical success（保持成功）** 必须同时满足：

1. 在 episode 中至少一次到达任务目标；
2. episode 结束时物体没有相对倾斜手掌掉出 `fallDistance`；
3. 因此，达到过目标后在时间上限 timeout 并不自动失败；只要物体仍被保持，就计为成功。

诊断默认会评估 `128` 个 episode / `128` 个环境；其中 `retained_success_rate >= 0.60` 是参考质量指标。block/egg/pen 的第二阶段会单独报告每种物体的保持成功率。它不会改变 `reward_only` 模式的晋级决定，也不会因为 evaluator 异常中断课程。

## 4. 多张 CUDA 卡、worker 与环境数设置

配置都在下面两个 manifest 的 `training` 段中：

- `curricula/shadowhand18_tilt_42.yaml`
- `curricula/wujihand_fixed_tilt_42.yaml`

`gpu_ids` 是唯一需要手动列出的设备列表。launcher 每个可用 GPU 最多启动一个独立训练进程；`workers` 由 `gpu_ids` 的长度自动得出，不需要也不应再手工设置旧的 `workers` 字段。例如只允许使用物理 GPU 0 和 2：

```yaml
training:
  gpu_ids: [0, 2]
  min_free_memory_mb: 12000
```

这会最多同时运行两个 worker，并分别给子进程设置 `CUDA_VISIBLE_DEVICES=0` 和 `CUDA_VISIBLE_DEVICES=2`。每个子进程内部仍使用自己的 `cuda:0`，所以不需要把 `sim_device` 或 `rl_device` 改成 `cuda:2`。

launcher 启动前读取 `nvidia-smi`。少于 `min_free_memory_mb` 的已列 GPU 会被跳过；如果 `nvidia-smi` 在容器中不可用，launcher 保守地使用 manifest 顶层的 `num_envs` 与 `minibatch_size`，由 CUDA 在启动时给出最终错误。确保 `gpu_ids` 只包含当前允许使用的设备。

`resource_profiles` 根据每张 GPU 当前空闲显存独立选择 `num_envs` 和 `minibatch_size`。所有 profile 都满足：

```text
(num_envs * horizon_length) % minibatch_size == 0
```

例如可以按实际机器调整为：

```yaml
training:
  num_envs: 10240            # nvidia-smi 不可用时的回退值
  minibatch_size: 10240
  horizon_length: 8
  gpu_ids: [0, 2, 3]
  min_free_memory_mb: 10000
  resource_profiles:
    - {min_free_memory_mb: 40000, num_envs: 20480, minibatch_size: 20480}
    - {min_free_memory_mb: 20000, num_envs: 10240, minibatch_size: 10240}
    - {min_free_memory_mb: 10000, num_envs: 5120, minibatch_size: 5120}
```

不要在同一块 GPU 上再额外手动启动 curriculum worker；该 launcher 的设计就是“一张列出的可用 GPU 对应一个 worker”。先查看实际将使用的 checkpoint、GPU profile 与 108 条方向：

```bash
python scripts/run_shadowhand18_tilt_curriculum.py --inspect
python scripts/run_wujihand_fixed_tilt_curriculum.py --inspect
```

确认 `checkpoint_search_roots`、`gpu_ids` 和手工验证的 offset 后，再启动：

```bash
python scripts/run_shadowhand18_tilt_curriculum.py \
  --python=/home/srtp/anaconda3/envs/isaac/bin/python

python scripts/run_wujihand_fixed_tilt_curriculum.py \
  --python=/home/srtp/anaconda3/envs/isaac/bin/python
```

新的运行状态写到 `curriculum_runs/*_dense_v5/`。旧状态和旧 checkpoint 不会被覆盖；中断后再次运行同一命令即可恢复，必要时使用 `--retry timed_out` 只重跑 timeout 方向。

如果是旧版 launcher 留下的 `failed` 条目、且输出 checkpoint 已存在：当前默认的 `reward_only` launcher 重启时会自动读取 checkpoint；只要 reward `>2500` 就直接恢复成 `succeeded`，无需重训也无需物理验收。只有你将 manifest 改为物理验收模式后，才可用下面命令只重跑已有 checkpoint 的物理验收，并原地更新同一个 `state.json`（会使用列出的 GPU 并行验收）：

```bash
python scripts/run_shadowhand18_tilt_curriculum.py \
  --state-dir /tmp/shadowhand18tillted_tilt_42 \
  --recertify-failed
```

在物理验收模式下，通过验收的条目会变成 `succeeded` 并可立即作为后续 parent；未通过 60% 保持成功率的条目仍保持 `failed`，这时才根据需要使用 `--retry failed` 重新训练。
