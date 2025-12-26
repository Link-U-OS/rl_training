import numpy as np
import pandas as pd
import random
import os
import torch


class RaiseA2Trajectory:
    def __init__(self, commands_csv_path=None, states_csv_path=None):
        """
        Load trajectory data from CSV files

        Args:
            commands_csv_path: Path to joint commands CSV file
            states_csv_path: Path to joint states CSV file
        """
        if commands_csv_path is not None and states_csv_path is not None:
            # Load from specified CSV files
            self._load_from_csv(commands_csv_path, states_csv_path)
        else:
            # Try multiple possible default paths
            default_paths = [
                # Current directory
                (
                    "rosbag_analysis_output/cycle_joint_commands_serial.csv",
                    "rosbag_analysis_output/cycle_joint_states_serial.csv",
                ),
                # Trajectory subdirectory
                (
                    "trajectory/rosbag_analysis_output/cycle_joint_commands_serial.csv",
                    "trajectory/rosbag_analysis_output/cycle_joint_states_serial.csv",
                ),
                # Parent directory with trajectory
                (
                    "../trajectory/rosbag_analysis_output/cycle_joint_commands_serial.csv",
                    "../trajectory/rosbag_analysis_output/cycle_joint_states_serial.csv",
                ),
            ]

            found_default = False
            for commands_path, states_path in default_paths:
                if os.path.exists(commands_path) and os.path.exists(states_path):
                    print(f"Loading default CSV trajectory from {commands_path} and {states_path}")
                    self._load_from_csv(commands_path, states_path)
                    found_default = True
                    break

            if not found_default:
                raise ValueError(
                    "Please provide CSV paths. Searched for default files in:\n"
                    + "\n".join([f"  - {cmd}, {state}" for cmd, state in default_paths])
                )

    def _load_from_csv(self, commands_csv_path, states_csv_path):
        """Load trajectory data from CSV files"""

        # Load CSV data
        commands_df = pd.read_csv(commands_csv_path)
        states_df = pd.read_csv(states_csv_path)

        # Clean column names
        commands_df.columns = commands_df.columns.str.strip()
        states_df.columns = states_df.columns.str.strip()

        # Extract time (use states timestamp as primary)
        self.time = states_df["timestamp"].values

        # Automatically detect joint names from states columns
        joint_names = []
        for col in states_df.columns:
            if col.endswith("_position"):
                joint_name = col.replace("_position", "")
                joint_names.append(joint_name)

        joint_names.sort()  # Ensure consistent ordering
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

        self.state_pos = np.array(state_pos_data).T  # Shape: (timesteps, n_joints)

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
        print(f"  States - pos: {self.state_pos.shape}, vel: {self.state_vel.shape}, effort: {self.state_eff.shape}")
        print(f"  Commands - pos: {self.cmd_pos.shape}, vel: {self.cmd_vel.shape}, effort: {self.cmd_eff.shape}")

    def _get_indices(self, t):
        """Get indices for time t, supporting both scalar and tensor inputs"""
        tmax = self.time[-1] - self.time[0]

        # Handle both torch tensors and numpy arrays/scalars
        if isinstance(t, torch.Tensor):
            t_norm = (t % tmax) / tmax
            indices = (t_norm * len(self.time)).long()
            # Clamp indices to valid range
            indices = torch.clamp(indices, 0, len(self.time) - 1)
            return indices
        else:
            # Handle numpy scalars/arrays
            t_norm = (t % tmax) / tmax
            indices = np.clip((t_norm * len(self.time)).astype(int), 0, len(self.time) - 1)
            return indices

    def state(self, t):
        """Get actual joint state (position, velocity, effort) at time t

        Args:
            t: Time value(s), can be scalar, numpy array, or torch tensor

        Returns:
            tuple: (positions, velocities, efforts) with shapes matching input
        """
        indices = self._get_indices(t)

        if isinstance(t, torch.Tensor):
            # Convert numpy arrays to torch tensors and index
            state_pos_tensor = torch.from_numpy(self.state_pos).float().to(t.device).requires_grad_(False)
            state_vel_tensor = torch.from_numpy(self.state_vel).float().to(t.device).requires_grad_(False)
            state_eff_tensor = torch.from_numpy(self.state_eff).float().to(t.device).requires_grad_(False)

            return (
                state_pos_tensor[indices],
                state_vel_tensor[indices],
                state_eff_tensor[indices],
            )
        else:
            # Handle numpy indexing
            return (
                self.state_pos[indices],
                self.state_vel[indices],
                self.state_eff[indices],
            )

    def action(self, t):
        """Get commanded joint action (position, velocity, effort) at time t

        Args:
            t: Time value(s), can be scalar, numpy array, or torch tensor

        Returns:
            tuple: (positions, velocities, efforts) with shapes matching input
        """
        indices = self._get_indices(t)

        if isinstance(t, torch.Tensor):
            # Convert numpy arrays to torch tensors and index
            cmd_pos_tensor = torch.from_numpy(self.cmd_pos).float().to(t.device).requires_grad_(False)
            cmd_vel_tensor = torch.from_numpy(self.cmd_vel).float().to(t.device).requires_grad_(False)
            cmd_eff_tensor = torch.from_numpy(self.cmd_eff).float().to(t.device).requires_grad_(False)

            return (
                cmd_pos_tensor[indices],
                cmd_vel_tensor[indices],
                cmd_eff_tensor[indices],
            )
        else:
            # Handle numpy indexing
            return (self.cmd_pos[indices], self.cmd_vel[indices], self.cmd_eff[indices])

    def effort(self, t):
        """Get actual joint effort at time t

        Args:
            t: Time value(s), can be scalar, numpy array, or torch tensor

        Returns:
            effort values with shape matching input
        """
        indices = self._get_indices(t)

        if isinstance(t, torch.Tensor):
            # Convert numpy array to torch tensor and index
            state_eff_tensor = torch.from_numpy(self.state_eff).float().to(t.device).requires_grad_(False)
            return state_eff_tensor[indices]
        else:
            # Handle numpy indexing
            return self.state_eff[indices]

    def sample(self):
        """Sample a random time step with state data"""
        i = random.randrange(len(self.time))
        return (self.time[i], self.state_pos[i], self.state_vel[i])
