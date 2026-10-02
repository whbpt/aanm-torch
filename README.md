# aanm-torch

AANM(Adaptive ANM)构象形变路径与分子机器 morph 轨迹工具 ——
`designer/machine/scripts/` 下 ProDy 版本的 PyTorch 重写,计算全部跑在 torch 张量上,
有 CUDA 的机器自动用 GPU(`--device cuda`),Mac 上可用 MPS,否则 CPU。

**支持蛋白质、DNA/RNA 及其混合复合物**:每个残基自动选一个网络节点
(氨基酸取 Cα,核苷酸取 P),主链解析同时保留蛋白 N/CA/C/O 与核酸磷酸骨架
P/O5′/C5′/C4′/C3′/O3′,cutoff 默认按节点间距自动建议(蛋白 ~15 Å,核酸 ~24 Å),
侧链/碱基可用 `sidechains: true` 刚体跟随。剪接体、核糖体类蛋白-RNA 混合体系
直接用 `"network": "assembly"` 一个弹性网络跑。

## 结构

```
aanm_torch/
  __init__.py    # 设备选择(cuda > mps > cpu;MPS 无 float64 时自动回落 CPU)
  pdbio.py       # PDB 解析(主链或全原子,蛋白+核酸)/多模型写出
  geom.py        # Kabsch(单个+批量)、旋转插值、RMSD(torch)
  anm.py         # ANM:稠密 Hessian、成对键列表、稀疏优先求解器、断链检测
  adaptive.py    # Adaptive ANM:oneway / alternating / 多转换批量(共享步进核)
  clash.py       # 精确逐对 clash 计数(支持按残基组跳过共价近邻)
  backbone.py    # CA/P 节点路径 -> 全原子重建(同链邻居、中点混合、侧链)
  saxs.py        # 可微 Debye 散射曲线(O(N^2) 成对核,GPU/批量)
  minaction.py   # 最小作用量路径:两端弹性势上的 Onsager-Machlup 作用量, LBFGS
  pathway.py     # ANMPathway 风格双向路径 + 逐步软化弛豫(纯软模式形变)
  clustenm.py    # ClustENM 风格系综:软模式热激发采样 + clash 过滤 + k-means
  cryofit.py     # NMFF 风格 cryo-EM 柔性拟合:极简 MRC 读写 + 可微高斯密度
                 # + 实空间相关系数,ANM 模式系数 autograd 优化
scripts/
  aanm_path.py            # 两状态间 AANM 路径 + 指标
  make_aanm_trajectory.py # 多状态循环轨迹生成
  saxs_curve.py           # PDB -> SAXS 曲线(.dat)
  cryofit.py              # PDB + MRC -> 柔性拟合结构
tests/                    # 59 个 unittest 用例 + ProDy 对比
```

### 经典工具的 torch 重写(设计取舍,诚实说明)

`saxs` / `minaction` / `pathway` / `clustenm` / `cryofit` 是五个经典工具的
torch 化,共同点是**全程可微 + GPU/批量**;相对原版的简化都经过测试验证并
在此声明:

- **saxs**:常数形状因子(电子数)而非 q 依赖的 Cromer-Mann + 溶剂排除——
  适合相对曲线与可微拟合,不适合绝对强度对标实验。两原子解析解、naive
  循环逐点一致、有限差分梯度一致。
- **minaction**:Onsager-Machlup 离散作用量 + 两端 ANM 势,势能项按端点
  能量归一(作用量绝对单位任意,只做路径间相对比较);与 Franklin et al.
  2007 服务器的常数约定不同。LBFGS 优化,作用量显著低于线性插值。
- **pathway**:ANMPathway.o 的双向自适应步进保留;其"切换势垒上的过渡态
  搜索"简化为**逐步软化弛豫**——每帧相对前一帧的位移严格投影到当前网络
  最软模式子空间(可精确检验,测试断言刚性能量占比 < 1e-8);汇入固定端点
  的收尾步豁免(锚点不可重投影)。
- **clustenm**:原版的力场能量极小化弛豫替换为"只激发软模式(不引入局部
  应变)+ clash 过滤";采样振幅以 Å 为尺度无关参数。无偏 ClustENM 做的是
  种子附近的平衡涨落,不做远距离盆地间过渡(那需要 adaptive/pathway 引导)。
- **cryofit**:极简 MRC(仅 mode-2 float32);模式系数以 RMSD-Å 为单位。
  已验证其工作区(软模式主导的构象变化可精确恢复,CCC→1.0、RMSD→0);
  也验证了其边界——目标形变不在软模式子空间时(合成体系投影仅 ~0.4),
  CCC 仍可通过非真实形变上升,**不能把模式基表达不了的变化拟合回去**,这是
  NMFF 方法的固有限制,使用时须自证形变的软模式覆盖度。

## 用法

```bash
# 两状态 AANM 路径
python scripts/aanm_path.py E1_2Ca_aligned.pdb E1_ATP_aligned.pdb 40 out_prefix \
    [--device auto|cuda|mps|cpu] [--dtype float64|float32] \
    [--sidechains] [--blend 0.2] [--solver auto] [--cutoff 15] [--gamma 1] \
    [--n-modes 10] [--f 0.3] [--fmin-max 0.6] [--target-rmsd 1.0] \
    [--min-rmsd-diff 0.05] [--aligned] [--quiet] [--allow-empty]

# 轨迹生成,config JSON 与原脚本兼容,新增键均为可选
python scripts/make_aanm_trajectory.py --config machines/serca/config_aanm.json \
    [--device auto] [--dtype float64] [--quiet] [--allow-empty]
```

config 的 `aanm` 段新增键(全部可选,默认保持与原脚本行为兼容):

| 键 | 默认 | 含义 |
|---|---|---|
| `network` | `"anchor"` | `"anchor"`:anchor 链 AANM,其余链经 site transforms 复制其重建主链(同源寡聚体假设各链序列相同);`"assembly"`:两状态共有链的全部 CA 拼成一个弹性网络,链间相对运动(转动/呼吸)直接来自低频模式,每条链各自重建 |
| `mode` | `"oneway"` | `"alternating"`:两端交替向中间形变(ProDy AANM_ALTERNATING),路径包含两个精确端点 |
| `sidechains` | `false` | 顶层键。解析全原子,重建整个残基(侧链刚体跟随残基局部系) |
| `blend` | `0.2` | 重建时 A/B 端点源在中点附近的线性混合窗口(路径分数;0 = 硬切换) |
| `solver` | `"auto"` | **稀疏优先**:`auto` ≤250 CA 用稠密 eigh(快且与 ProDy 逐位一致),之后优先 scipy 稀疏 `eigsh`(Hessian 从键列表稀疏组装,内存 O(边数)),scipy 不可用则 `lobpcg`;也可显式指定。实测 ~700 CA 处稀疏反超稠密;20000 CA(6 万自由度)单步 ~2 分钟;GroEL 14 聚体真实数据整条轨迹 15 s(`experiments/04_bench_sparse.py`) |
| `cutoff` / `gamma` / `fmin_max` / `target_rmsd` / `min_rmsd_diff` | 15 / 1 / 0.6 / 1.0 / 0.05 | AANM 参数透传 |
| `batch`(顶层) | `"auto"` | `"auto"` 在 CUDA 上把所有同尺寸 transition 的 Hessian 堆成批量、一次批量 eigh(锁步推进,提前收敛者跳过);`true` 强制、`false` 串行 |

设备与精度:
- `--dtype float64`(默认)与 ProDy 数值一致;CUDA 原生支持。
- `--dtype float32` 在 GPU 上更快;本数据逐步 RMSD 与 float64 一致到 ~1e-3 Å。
- MPS 不支持 float64(auto 回落 CPU)且无 eigh(逐 CT 回落 CPU 对角化)。
  推荐运行环境:本仓库 `.venv`(`uv venv .venv && uv pip install torch numpy scipy`,
  torch 2.x + 健康的 scipy 即可启用 `eigsh`)——Levin ligandmpnn 环境的
  scipy.sparse 已损坏(_propack dylib 加载失败),那里 `eigsh` 不可用、
  `auto` 只能落到 lobpcg。

## 与 ProDy 参考实现的数值验证

| 量 | 最大差异 |
|---|---|
| ANM Hessian(994 CA) | 2.5e-14 |
| 逐步 RMSD(10 步) | 6.3e-13 |
| 路径帧坐标 | 4.4e-12 Å |
| 完整轨迹 PDB(16 帧,v1.0 硬切换语义) | PDB 1e-3 Å 精度下逐位一致 |
| 合成体系 oneway(含停滞用例) | 1.8e-15(ProDy 同样停滞,属算法固有行为) |

v1.1 语义变化(默认 blend=0.2)会让 morph 中点附近的帧与 v1.0 有小幅差异,
clash 指标不变(SERCA mini 回归:同为 0 处 <1.6 Å clash)。

## 规模与速度(诚实的数字)

本机 Mac CPU 实测(`experiments/04_bench_sparse.py`,每次模式求解耗时):

| 体系规模 | 每次求解 | 说明 |
|---|---|---|
| 2,000 残基 | 2.2 s | 稠密 eigh 要 14.4 s(6.4 倍) |
| 5,000 残基 | 12 s | 稠密路线在此之上不可行 |
| 10,000 残基 | 34 s | 3 万自由度 |
| 20,000 残基 | ~2 min | 6 万自由度,内存 O(键数) |

整条轨迹换算:SERCA(994 残基)秒级;GroEL 十四聚体整体网络实测 15 秒;
核糖体级(约 5 千网络节点)分钟级一步、小时级整条;2 万残基约 1.5 小时。
要点:内存不再是瓶颈(这是与旧版"2500 残基上限"的本质区别);时间单步
始终是秒到分钟级。eigsh 走 CPU、GPU 加速的是多状态批量 eigh,两者互补;
百倍规模(病毒衣壳 25 万残基)需要 RTB 块降维,尚未实现。

## 相对原 ProDy 方案的全部改进点

**计算引擎:ProDy → PyTorch,数值不降级**
- ProDy 依赖完全去除(PDB 解析、几何、求解全自写);CUDA 一等公民,Mac MPS/CPU 自适应;
- 非近似重写:Hessian 与 ProDy 差 2.5e-14、逐步 RMSD ~1e-13、轨迹 PDB 逐位一致,由 ProDy 2.4.1 对照测试(子进程)长期守住;
- 多状态转换 Hessian 堆成批量、一次对角化(GPU 利用率大幅提升),测试保证与串行逐位一致。

**规模:2500 残基上限 → 2 万残基可行**
- 稀疏优先求解(≤250 CA 稠密 eigh,之上 scipy `eigsh` shift-invert,内存 O(边数);无 scipy 则 GPU `lobpcg` 兜底)。

**科学质量**
- assembly 整体网络:寡聚体不再"anchor 链算完、其它链刚体复制",链间相对运动来自低频模式;
- alternating 双向步进(可选),对单方向停滞的体系是解药;
- 重建三处修正:邻居限同链内(修跨链借邻居 bug)、端点源中点平滑混合(消原子跳变)、侧链刚体跟随;
- clash 计数修正:原网格版系统性漏检跨格近邻且误计肽键,现为精确逐对 + 残基组共价排除;
- 空 MODEL 输出保护(老代码曾默默写出空帧)。

**分子类型:蛋白、DNA/RNA 及混合复合物**
- 每残基自动选网络节点(氨基酸 Cα / 核苷酸 P),混合体系一次解析进同一网络,cutoff 按节点间距自动建议(蛋白 ~15 Å、RNA ~24 Å)。

**工程**
- 零测试 → 40 个单元测试(几何、Hessian 手算值、求解器互检、重建连续性、批量-串行一致、核酸、端到端轨迹、ProDy 对比);
- 全参数可配置(cutoff/gamma/Fmin/target_rmsd/solver/blend/sidechains/network/mode/batch),逐帧进度输出,dwell 重复帧扫描去重;
- 性能修复:重建预处理 O(N²)→O(N)、批量 SVD 重建、断链检测分块。

细节差异(与原始逐条对照):
1. **clash 计数是精确的逐对统计**:原网格版只比较同一 1 Å 格内的原子,跨格近邻对
   全部漏掉。共价近邻通过残基组(全原子模式)或 4 原子列表偏移(主链模式)排除——
   肽键 C(i)–N(i+1)(1.33 Å)不再被计入。CA 层扫描(3.5 Å,skip=2)语义不变。
2. **重建的邻居约束在同链内**:原版按全局排序取邻居,A 链末端残基会借 B 链 CA 定向;
   每条链的首尾残基(只有 2 个相邻 CA)按原逻辑跳过。
3. **端点中点混合**消除 N/C/O 在路径中点的跳变(可用 blend=0 关闭)。
4. **侧链刚体跟随**:`--sidechains` / config `sidechains: true`。中点附近侧链
   clash 属正常近似误差(端点帧干净)。
5. Hessian 分块构建,(N,N,3,3) 对块张量不全量物化;大体系走稀疏/lobpcg 路径。
6. dwell 重复帧 clash 扫描去重(每块只扫首帧);空帧默认报错(--allow-empty 放行)。
7. 批量 eigh 见上表 `batch`;非稠密求解器自动回退串行。

## 测试

```bash
python -m unittest discover -s tests -t .     # 33 个用例,~2 分钟
```

覆盖:Kabsch/旋转插值、Hessian 解析手算值与对称性、三种求解器互相校验、
clash 列表/分组语义、重建(首尾跳过、同链邻居、中点混合连续性、侧链)、
oneway/alternating/批量与串行逐位一致、轨迹端到端(anchor/assembly/侧链/交替/
空帧保护)、ProDy 对比(子进程跑 `.venv-morph` 的 ProDy 2.4.1,SERCA 5 步
RMSD 差 ≤1e-9;ProDy 或数据缺失时自动跳过)。

## 依赖

`torch >= 2.0`、`numpy`;大体系稀疏求解可选 `scipy`。PDB 解析与写出不依赖 ProDy。
