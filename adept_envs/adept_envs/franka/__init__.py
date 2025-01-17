#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gym.envs.registration import register

# Relax the robot
register(
    id='kitchen_relax-v1',
    entry_point='adept_envs.franka.kitchen_multitask_v0:KitchenTaskRelaxV1',
    kwargs={"cam_height": 1920, "cam_width": 2560},
    max_episode_steps=280,
)

register(
    id='kitchen_relax-v2',
    entry_point='adept_envs.franka.kitchen_multitask_v0:KitchenTaskRelaxV2',
    kwargs={"cam_height": 1920, "cam_width": 2560},
    max_episode_steps=280,
)

register(
    id='kitchen_relax_model-v1',
    entry_point='adept_envs.franka.kitchen_multitask_v0:KitchenTaskRelaxModelV1',
    kwargs={"model_filename": 'franka_kitchen_jntpos_act_ab.xml', \
        "cam_height": 1920, "cam_width": 2560},
    max_episode_steps=280,
)

register(
    id='kitchen_kettle-v0',
    entry_point='adept_envs.franka.kitchen_multitask_v0:KitchenKettleV0',
    kwargs={"model_filename": 'franka_kitchen_jntpos_act_ab.xml', \
        "cam_height": 1920, "cam_width": 2560},
    max_episode_steps=280,
)

register(
    id='kitchen_microwave-v0',
    entry_point='adept_envs.franka.kitchen_multitask_v0:KitchenMicrowaveV0',
    kwargs={"model_filename": 'franka_kitchen_jntpos_act_ab.xml', \
        "cam_height": 1920, "cam_width": 2560},
    max_episode_steps=280,
)