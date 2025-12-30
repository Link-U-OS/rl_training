[English](README.md) | 中文

## 简介

远征 A2 是由智元研发并开源的模块化、高自由度人形机器人，本工程为远征 A2 所使用的强化学习训练代码，可配合远征 A2 配套的部署框架进行真机和仿真的行走调试，或导入其他机器人模型进行训练。

![](doc/a2.jpg)

## 代码运行

### 安装依赖
1. 创建一个新的python3.8虚拟环境:
   - `conda create -n myenv python=3.8`.
   - 激活虚拟环境 `conda activate myenv`.
2. 安装 Isaac Gym:
   - 下载和安装 Isaac Gym Preview 4  https://developer.nvidia.com/isaac-gym.
   - `cd isaacgym/python && pip install -e .`
   - 运行示例 `cd examples && python 1080_balls_of_solitude.py`
   - 如需故障排查，请查阅 `isaacgym/docs/index.html`
3. 安装训练代码依赖：
   - 克隆此仓库.
   - `pip install -e .`

### 使用

#### Train:

```python humanoid/scripts/train.py --task=a2_dh_stand --run_name=<run_name> --headless```
- 训练好的模型会存`/logs/<experiment_name>/exported_data/<date_time><run_name>/model_<iteration>.pt` 其中 `<experiment_name>` 在config文件中定义。

#### Play:

```python humanoid/scripts/play.py --task=a2_dh_stand --load_run=<date_time><run_name>```

#### 生成jit模型:

``` python humanoid/scripts/export_policy_dh.py --task=a2_dh_stand --load_run=<date_time><run_name>  ```
- jit模型会存`/logs/exported_policies/<date_time>`

#### 生成onnx模型:

``` python humanoid/scripts/export_onnx_dh.py --task=a2_dh_stand --load_run=<date_time>  ```
- onnx模型会存`/logs/exported_policies/<date_time>`

#### 查看训练日志:

``tensorboard --logdir logs``

#### 参数说明：
- task: 任务名称
- resume: 从检查点恢复训练
- experiment_name: 要运行或加载的实验名称
- run_name: 运行名称
- load_run: 当 resume=True 时要加载的运行名称；若为 -1，则加载最近一次的运行
- checkpoint: 已保存的模型检查点编号；若为 -1，则加载最新的检查点
- num_envs: 要创建的环境数量
- seed: 随机种子
- max_iterations: 最大训练迭代次数

### 添加新环境

1. 在 `envs/`目录下创建一个新文件夹，在新文件夹下创建一个配置文件`<your_env>_config.py`和环境文件`<your_env>_env.py`，这两个文件要分别继承`LeggedRobotCfg`和`LeggedRobot`。

2. 将新机器的urdf, mesh, mjcf放到 `resources/`文件夹下。
- 在`<your_env>_config.py`里配置新机器的urdf path，PD gain，body name, default_joint_angles, experiment_name 等参数。

3. 在`humanoid/envs/__init__.py`里注册你的新机器人。

### sim2sim
请使用部署框架进行sim2sim验证。
## 目录结构
```
.
|— humanoid           # 主要代码目录
|  |—algo             # 算法目录
|  |—envs             # 环境目录
|  |—scripts          # 脚本目录
|  |—utilis           # 工具、功能目录
|— logs               # 模型目录
|— resources          # 资源库
|  |— robots          # 机器人urdf, mjcf, mesh
|— README.md          # 说明文档
```



> 参考项目:
>
> * [GitHub - leggedrobotics/legged_gym: Isaac Gym Environments for Legged Robots](https://github.com/leggedrobotics/legged_gym)
> * [GitHub - leggedrobotics/rsl_rl: Fast and simple implementation of RL algorithms, designed to run fully on GPU.](https://github.com/leggedrobotics/rsl_rl)
> * [GitHub - roboterax/humanoid-gym: Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer https://arxiv.org/abs/2404.05695](https://github.com/roboterax/humanoid-gym)
> * [Github - AgibotTedch/agibot_x1_tain: The reinforcement learning training code for AgiBot X1.](https://github.com/AgibotTech/agibot_x1_train)
