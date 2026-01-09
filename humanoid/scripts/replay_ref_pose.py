import time
import os
import re
import numpy as np
import pandas as pd

import mujoco
import mujoco.viewer

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs.a2.trajectory import RaiseA2Trajectory

import torch


def load_mjcf_safe(mjcf_path):
    abs_path = os.path.abspath(mjcf_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"MJCF file not found: {abs_path}")
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
    print(f"loading model: {mjcf_path}")
    try:
        model = load_mjcf_safe(mjcf_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return
    print("Model loaded successfully.")

    if not os.path.exists(state_csv):
        print(f"Error: Trajectory file {state_csv} not found")
        return

    traj = RaiseA2Trajectory(cmd_csv, state_csv)

    states_df = pd.read_csv(state_csv)
    states_df.columns = states_df.columns.str.strip()

    joint_names = sorted(
        col.replace("_position", "")
        for col in states_df.columns
        if col.endswith("_position")
    )

    print(f"Detected {len(joint_names)} joints from CSV")

    mj_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]

    col_idx_to_qpos_addr = {}
    col_idx_to_qvel_addr = {}
    col_idx_to_qfrc_addr = {}

    print("\nMatching joint data...")
    matched = 0
    for i, csv_jname in enumerate(joint_names):
        found = False
        for mj_jname in mj_joint_names:
            if mj_jname and csv_jname in mj_jname:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_jname)

                # position jnt_qposadr , vel/frc jnt_dofadr
                qpos_addr = model.jnt_qposadr[jid]
                dof_addr = model.jnt_dofadr[jid]

                col_idx_to_qpos_addr[i] = qpos_addr
                col_idx_to_qvel_addr[i] = dof_addr
                col_idx_to_qfrc_addr[i] = dof_addr

                found = True
                matched += 1
                break
        if not found:
            print(f"  [Warning] Trajectory joint '{csv_jname}' not found in the model")

    print(f"  [Warning] Trajectory joint '{csv_jname}' not found in the model")

    sim_duration = traj.time[-1] - traj.time[0]
    dt = 0.005

    print(f"  [Warning] Trajectory joint '{csv_jname}' not found in the model")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            now = time.time()
            sim_t = now - start_time

            if sim_t > sim_duration:
                start_time = now
                sim_t = 0

            try:
                positions, velocities, efforts = traj.state(sim_t)

                for i, qpos_addr in col_idx_to_qpos_addr.items():
                    data.qpos[qpos_addr] = positions[i]

                for i, qvel_addr in col_idx_to_qvel_addr.items():
                    data.qvel[qvel_addr] = velocities[i]

                for i, qfrc_addr in col_idx_to_qfrc_addr.items():
                    data.qfrc_applied[qfrc_addr] = efforts[i]

                mujoco.mj_forward(model, data)
                viewer.sync()

                time.sleep(dt)

            except Exception as e:
                print(f"Replay error: {e}")
                break


if __name__ == "__main__":
    mjcf_file = "resources/robots/agibot_a2/mjcf/agibot_a2_dof12.mjcf"
    csv_dir = "trajectory/rosbag_analysis_output"

    cmd_csv = os.path.join(csv_dir, "cycle_joint_commands_serial.csv")
    state_csv = os.path.join(csv_dir, "cycle_joint_states_serial.csv")

    run_replay(mjcf_file, cmd_csv, state_csv)
