# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-FileCopyrightText: Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2024, AgiBot Inc. All rights reserved.

from humanoid import LEGGED_GYM_ROOT_DIR
import os
import copy
import re
from humanoid.envs import *
from humanoid.utils import get_args, task_registry
from humanoid.utils.helpers import get_load_path, class_to_dict
import torch
from humanoid.algo.ppo import ActorCriticDH


class ExportedDH(torch.nn.Module):
    def __init__(
        self,
        actor,
        long_history,
        state_estimator,
        num_short_obs,
        in_channels,
        num_proprio_obs,
    ):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.long_history = copy.deepcopy(long_history).cpu()
        self.state_estimator = copy.deepcopy(state_estimator).cpu()
        self.num_short_obs = num_short_obs
        self.in_channels = in_channels
        self.num_proprio_obs = num_proprio_obs

    def forward(self, observations):
        short_history = observations[..., -self.num_short_obs :]
        es_vel = self.state_estimator(short_history)
        compressed_long_history = self.long_history(observations.view(-1, self.in_channels, self.num_proprio_obs))
        actor_obs = torch.cat((short_history, es_vel, compressed_long_history), dim=-1)
        actions_mean = self.actor(actor_obs)
        return actions_mean


def export_onnx(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    train_cfg_dict = class_to_dict(train_cfg)
    policy_cfg = train_cfg_dict["policy"]
    num_critic_obs = env_cfg.env.num_privileged_obs
    if env_cfg.terrain.measure_heights:
        num_critic_obs = env_cfg.env.c_frame_stack * (
            env_cfg.env.single_num_privileged_obs + env_cfg.terrain.num_height
        )
    num_short_obs = env_cfg.env.short_frame_stack * env_cfg.env.num_single_obs
    actor_critic_class = eval(train_cfg_dict["runner"]["policy_class_name"])
    actor_critic: ActorCriticDH = actor_critic_class(
        num_short_obs,
        env_cfg.env.num_single_obs,
        num_critic_obs,
        env_cfg.env.num_actions,
        **policy_cfg,
    )

    # load policy from exported_data
    log_root_encoder = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported_data")

    # Extract run_name for file naming
    run_name_for_file = None
    original_load_run = args.load_run

    # If load_run is a run_name (not a full directory path), find matching directory
    if args.load_run != -1 and isinstance(args.load_run, str):
        if not os.path.exists(os.path.join(log_root_encoder, args.load_run)):
            # Try to find a directory that ends with the run_name
            if os.path.exists(log_root_encoder):
                runs = os.listdir(log_root_encoder)
                if "exported" in runs:
                    runs.remove("exported")
                matching_runs = [r for r in runs if r.endswith(args.load_run)]
                if matching_runs:
                    matching_runs.sort()  # Get the most recent one
                    args.load_run = matching_runs[-1]
                    print(f"Found matching directory: {args.load_run}")
                    # Extract run_name from directory name (e.g., 2025-10-30_22-00-09refpose1 -> refpose1)
                    run_name_for_file = args.load_run
                else:
                    raise FileNotFoundError(
                        f"No directory found ending with '{args.load_run}' in {log_root_encoder}\n"
                        f"Available directories: {runs}"
                    )
            else:
                raise FileNotFoundError(
                    f"Directory does not exist: {log_root_encoder}\n"
                    f"Please ensure the model has been trained and exported."
                )
        else:
            # Exact match, use the original run_name
            run_name_for_file = original_load_run
    elif args.load_run == -1:
        # Will use the last run, extract run_name from the directory name
        if os.path.exists(log_root_encoder):
            runs = os.listdir(log_root_encoder)
            if "exported" in runs:
                runs.remove("exported")
            if runs:
                runs.sort()
                run_name_for_file = runs[-1]

    model_path = get_load_path(log_root_encoder, load_run=args.load_run, checkpoint=args.checkpoint)
    print("Load model from:", model_path)

    # Extract checkpoint number from model path (e.g., model_20000.pt -> 20000)
    checkpoint_num = None
    model_filename = os.path.basename(model_path)
    match = re.search(r"model_(\d+)\.pt", model_filename)
    if match:
        checkpoint_num = match.group(1)

    # Extract run_name from directory name if not already set
    if run_name_for_file:
        # Extract the run_name part (everything after the timestamp pattern)
        # Pattern: YYYY-MM-DD_HH-MM-SSrun_name
        match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(.+)", run_name_for_file)
        if match:
            run_name_for_file = match.group(1)

    loaded_dict = torch.load(model_path)
    actor_critic.load_state_dict(loaded_dict["model_state_dict"])

    # create exported policy model
    exported_policy = ExportedDH(
        actor_critic.actor,
        actor_critic.long_history,
        actor_critic.state_estimator,
        num_short_obs,
        policy_cfg["in_channels"],
        env_cfg.env.num_single_obs,
    )
    exported_policy.eval()
    exported_policy.to("cpu")

    # export onnx to unified directory (no date prefix)
    root_path = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name, "exported_onnx")
    os.makedirs(root_path, exist_ok=True)

    # Generate filename with run_name and checkpoint
    if run_name_for_file and checkpoint_num:
        file_name = f"{run_name_for_file}_ckpt{checkpoint_num}.onnx"
    elif run_name_for_file:
        file_name = f"{run_name_for_file}.onnx"
    elif checkpoint_num:
        file_name = f"{args.task.split('_')[0]}_ckpt{checkpoint_num}.onnx"
    else:
        file_name = f"{args.task.split('_')[0]}_policy.onnx"

    path = os.path.join(root_path, file_name)
    example_input = torch.randn(1, env_cfg.env.num_observations)

    # export onnx model
    torch.onnx.export(
        exported_policy,  # model
        example_input,  # model example input
        path,  # model output path
        export_params=True,  # export model params
        opset_version=11,  # ONNX opset version
        do_constant_folding=True,  # optimize constant variable folding
        input_names=["input"],  # model input name
        output_names=["output"],  # model output name
    )
    print("Export onnx model to: ", path)


if __name__ == "__main__":
    args = get_args()
    # Support both --run_name and --load_run
    if args.load_run is None:
        if args.run_name is not None:
            args.load_run = args.run_name
        else:
            args.load_run = -1
    if args.checkpoint is None:
        args.checkpoint = -1
    export_onnx(args)
