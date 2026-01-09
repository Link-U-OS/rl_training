import time
import os
import re
import numpy as np
import pandas as pd
import torch
import mujoco
import mujoco.viewer


class RaiseA2Trajectory:
    def __init__(self, commands_csv_path, states_csv_path):
        self._load_from_csv(commands_csv_path, states_csv_path)

    def _load_from_csv(self, commands_csv_path, states_csv_path):
        commands_df = pd.read_csv(commands_csv_path)
        states_df = pd.read_csv(states_csv_path)

        commands_df.columns = commands_df.columns.str.strip()
        states_df.columns = states_df.columns.str.strip()

        self.time = states_df["timestamp"].values

        joint_names = []
        for col in states_df.columns:
            if col.endswith("_position"):
                joint_name = col.replace("_position", "")
                joint_names.append(joint_name)

        joint_names.sort()
        self.joint_names = joint_names

        print(
            f"Detected {len(joint_names)} joints: {joint_names[:5]}..."
            if len(joint_names) > 5
            else f"Detected joints: {joint_names}"
        )

        # Extract positions from states (actual joint positions)
        state_pos_data = []
        for joint in joint_names:
            pos_col = f"{joint}_position"
            if pos_col in states_df.columns:
                state_pos_data.append(states_df[pos_col].values)
            else:
                print(f"Warning: {pos_col} not found in states data")
                state_pos_data.append(np.zeros(len(states_df)))
        # 转置前结构 joint1 所有时间 joint2 所有时间
        self.state_pos = np.array(
            state_pos_data
        ).T  # 转置后结构 Shape: (timesteps, n_joints)

        # Extract velocities from states (actual joint velocities)
        state_vel_data = []
        for joint in joint_names:
            vel_col = f"{joint}_velocity"
            if vel_col in states_df.columns:
                state_vel_data.append(states_df[vel_col].values)
            else:
                print(f"Warning: {vel_col} not found in states data")
                state_vel_data.append(np.zeros(len(states_df)))

        self.state_vel = np.array(state_vel_data).T  # Shape: (timesteps, n_joints)

        # Extract efforts from states (actual joint efforts)
        state_effort_data = []
        for joint in joint_names:
            effort_col = f"{joint}_effort"
            if effort_col in states_df.columns:
                state_effort_data.append(states_df[effort_col].values)
            else:
                print(f"Warning: {effort_col} not found in states data")
                state_effort_data.append(np.zeros(len(states_df)))

        self.state_eff = np.array(state_effort_data).T  # Shape: (timesteps, n_joints)

        # Extract motor positions from commands (desired positions)
        cmd_pos_data = []
        for joint in joint_names:
            pos_col = f"{joint}_position"
            if pos_col in commands_df.columns:
                cmd_pos_data.append(commands_df[pos_col].values)
            else:
                print(f"Warning: {pos_col} not found in commands data")
                cmd_pos_data.append(np.zeros(len(commands_df)))

        self.cmd_pos = np.array(cmd_pos_data).T  # Shape: (timesteps, n_joints)

        # Extract motor velocities from commands (desired velocities)
        cmd_vel_data = []
        for joint in joint_names:
            vel_col = f"{joint}_velocity"
            if vel_col in commands_df.columns:
                cmd_vel_data.append(commands_df[vel_col].values)
            else:
                print(f"Warning: {vel_col} not found in commands data")
                cmd_vel_data.append(np.zeros(len(commands_df)))

        self.cmd_vel = np.array(cmd_vel_data).T  # Shape: (timesteps, n_joints)

        # Extract motor efforts from commands (desired efforts)
        cmd_effort_data = []
        for joint in joint_names:
            effort_col = f"{joint}_effort"
            if effort_col in commands_df.columns:
                cmd_effort_data.append(commands_df[effort_col].values)
            else:
                print(f"Warning: {effort_col} not found in commands data, using zeros")
                cmd_effort_data.append(np.zeros(len(commands_df)))

        self.cmd_eff = np.array(cmd_effort_data).T  # Shape: (timesteps, n_joints)

        print("Loaded trajectory data:")
        print(f"  Time steps: {len(self.time)}")
        print(f"  Duration: {self.time[-1] - self.time[0]:.3f} seconds")
        print(f"  Joints: {len(joint_names)}")

    def _get_indices(self, t):
        """Get indices for time t"""
        tmax = self.time[-1] - self.time[0]
        if isinstance(t, torch.Tensor):
            # 取模运算 (%) 实现循环播放：如果 t 超过总时长，回到开头
            t_norm = (t % tmax) / tmax
            indices = (t_norm * len(self.time)).long()  # （0-1）->(0,length)
            indices = torch.clamp(indices, 0, len(self.time) - 1)
            return indices
        else:
            t_norm = (t % tmax) / tmax
            indices = np.clip(
                (t_norm * len(self.time)).astype(int), 0, len(self.time) - 1
            )
            return indices

    def state(self, t):
        """Get actual joint state (position, velocity, effort) at time t"""
        indices = self._get_indices(t)
        if isinstance(t, torch.Tensor):
            state_pos_tensor = torch.from_numpy(self.state_pos).float().to(t.device)
            state_vel_tensor = torch.from_numpy(self.state_vel).float().to(t.device)
            state_eff_tensor = torch.from_numpy(self.state_eff).float().to(t.device)
            return (
                state_pos_tensor[indices],
                state_vel_tensor[indices],
                state_eff_tensor[indices],
            )
        else:
            return (
                self.state_pos[indices],
                self.state_vel[indices],
                self.state_eff[indices],
            )

    def action(self, t):
        """Get commanded joint action"""
        indices = self._get_indices(t)
        if isinstance(t, torch.Tensor):
            cmd_pos_tensor = torch.from_numpy(self.cmd_pos).float().to(t.device)
            cmd_vel_tensor = torch.from_numpy(self.cmd_vel).float().to(t.device)
            cmd_eff_tensor = torch.from_numpy(self.cmd_eff).float().to(t.device)
            return (
                cmd_pos_tensor[indices],
                cmd_vel_tensor[indices],
                cmd_eff_tensor[indices],
            )
        else:
            return self.cmd_pos[indices], self.cmd_vel[indices], self.cmd_eff[indices]

    def effort(self, t):
        """Get actual joint effort only"""
        indices = self._get_indices(t)
        if isinstance(t, torch.Tensor):
            return torch.from_numpy(self.state_eff).float().to(t.device)[indices]
        else:
            return self.state_eff[indices]


def load_mjcf_safe(mjcf_path):
    abs_path = os.path.abspath(mjcf_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"找不到 MJCF 文件: {abs_path}")
    model_dir = os.path.dirname(abs_path)
    assets = {}
    with open(abs_path, "r") as f:
        xml_string = f.read()
    pattern = re.compile(r'(?:file|filename)=["\']([^"\']+)["\']')
    paths = pattern.findall(xml_string)
    for rel_path in paths:
        if rel_path in assets:
            continue
        asset_path = os.path.normpath(os.path.join(model_dir, rel_path))
        if os.path.exists(asset_path):
            with open(asset_path, "rb") as f:
                assets[rel_path] = f.read()
    return mujoco.MjModel.from_xml_string(xml_string, assets=assets)


def run_replay(mjcf_path, cmd_csv, state_csv):
    print(f"正在加载模型: {mjcf_path}")
    try:
        model = load_mjcf_safe(mjcf_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    print("模型加载成功.")

    if not os.path.exists(state_csv):
        print(f"错误: 找不到轨迹文件 {state_csv}")
        return

    traj = RaiseA2Trajectory(cmd_csv, state_csv)

    mj_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]

    col_idx_to_qpos_addr = {}
    col_idx_to_qvel_addr = {}
    col_idx_to_qfrc_addr = {}

    print("\n正在匹配关节数据...")
    matched = 0
    for i, csv_jname in enumerate(traj.joint_names):
        found = False
        for mj_jname in mj_joint_names:
            if mj_jname and csv_jname in mj_jname:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_jname)

                # 位置用 jnt_qposadr ，速度和力用 jnt_dofadr
                qpos_addr = model.jnt_qposadr[jid]
                dof_addr = model.jnt_dofadr[jid]

                col_idx_to_qpos_addr[i] = qpos_addr
                col_idx_to_qvel_addr[i] = dof_addr
                col_idx_to_qfrc_addr[i] = dof_addr

                found = True
                matched += 1
                break
        if not found:
            print(f"  [警告] 轨迹关节 '{csv_jname}' 未在模型中找到对应项")

    print(f"匹配完成: {matched}/{len(traj.joint_names)} 个关节")

    sim_duration = traj.time[-1] - traj.time[0]
    dt = 0.005

    print(f"\n启动 MuJoCo 查看器... (时长: {sim_duration:.2f}s)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            now = time.time()
            sim_t = now - start_time

            # 循环
            if sim_t > sim_duration:
                start_time = now
                sim_t = 0

            # 获取数据
            try:
                positions, velocities, efforts = traj.state(sim_t)

                for i, qpos_addr in col_idx_to_qpos_addr.items():
                    data.qpos[qpos_addr] = positions[i]

                for i, qvel_addr in col_idx_to_qvel_addr.items():
                    data.qvel[qvel_addr] = velocities[i]

                for i, qfrc_addr in col_idx_to_qfrc_addr.items():
                    data.qfrc_applied[qfrc_addr] = efforts[i]

                # 前向动力学
                mujoco.mj_forward(model, data)
                viewer.sync()

                # 帧率控制
                time.sleep(dt)

            except Exception as e:
                print(f"回放出错: {e}")
                break


if __name__ == "__main__":
    mjcf_file = "resources/robots/agibot_a2/mjcf/agibot_a2_dof12.mjcf"
    csv_dir = "trajectory/rosbag_analysis_output"

    cmd_csv = os.path.join(csv_dir, "cycle_joint_commands_serial.csv")
    state_csv = os.path.join(csv_dir, "cycle_joint_states_serial.csv")

    run_replay(mjcf_file, cmd_csv, state_csv)
