# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Newton simulation carb settings."""

import carb

from .solver_config import MuJoCoSolverConfig

""" List of existing settings"""


def reset_on_stop() -> bool:
    """Check whether stopping the timeline resets the simulation.

    Returns:
        True if the simulation resets when the timeline stops.
    """
    return carb.settings.get_settings().get_as_bool("/physics/resetOnStop") or False


""" List of newton settings"""
# don't forget to register them


def mujoco_solver_config() -> MuJoCoSolverConfig:
    """Read default MuJoCo options from Kit's user/application configuration.

    Explicit ``configure_newton()`` solver objects take precedence over these
    defaults. ``autoSize`` maps the existing optional capacities to ``None``.
    """
    settings = carb.settings.get_settings()
    path = "/exts/isaacsim.physics.newton/mujoco"
    options = {}
    for setting, field in (("useMujocoContacts", "use_mujoco_contacts"), ("includeSites", "include_sites")):
        value = settings.get(f"{path}/{setting}")
        if value is not None:
            options[field] = value
    if settings.get(f"{path}/autoSize"):
        options.update(njmax=None, nconmax=None)
    return MuJoCoSolverConfig(**options)
