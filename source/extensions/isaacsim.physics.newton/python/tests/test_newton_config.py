# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Verify public Newton runtime configuration behavior."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import carb
import numpy as np

import omni.kit.test
from isaacsim.physics.newton import (
    CollisionConfig,
    MuJoCoSolverConfig,
    NewtonStage,
    NewtonConfig,
    NewtonSolverOutputConfig,
    configure_newton,
    get_newton_config,
)
from isaacsim.physics.newton.impl import carb_config


class TestNewtonConfig(omni.kit.test.AsyncTestCase):
    """Tests for ``configure_newton()`` runtime configuration."""

    async def setUp(self) -> None:
        """Save the current Newton configuration."""
        super().setUp()
        self._saved_config = copy.deepcopy(get_newton_config()) or NewtonConfig()
        self._settings = carb.settings.get_settings()
        self._settings_path = "/exts/isaacsim.physics.newton/mujoco"
        self._saved_settings = self._settings.get(self._settings_path)

    async def tearDown(self) -> None:
        """Restore the previous Newton configuration."""
        configure_newton(self._saved_config)
        if self._saved_settings is None:
            self._settings.destroy_item(self._settings_path)
        else:
            self._settings.set(self._settings_path, self._saved_settings)
        super().tearDown()

    async def test_configure_newton_updates_solver_outputs(self) -> None:
        """Configure optional solver-computed outputs before initialization."""
        config = NewtonConfig(
            solver_outputs=NewtonSolverOutputConfig(
                contact_forces=False,
                link_incoming_joint_force=True,
            ),
        )
        configure_newton(config)

        live = get_newton_config()
        self.assertIsNotNone(live)
        self.assertFalse(live.solver_outputs.contact_forces)
        self.assertTrue(live.solver_outputs.link_incoming_joint_force)

    async def test_absent_mujoco_settings_preserve_defaults(self) -> None:
        self._settings.destroy_item(self._settings_path)
        config = carb_config.mujoco_solver_config()
        self.assertEqual(config, MuJoCoSolverConfig())

    async def test_mujoco_settings_map_to_existing_config(self) -> None:
        self._settings.set(
            self._settings_path, {"autoSize": True, "useMujocoContacts": True, "includeSites": True}
        )
        config = carb_config.mujoco_solver_config()
        self.assertIsNone(config.njmax)
        self.assertIsNone(config.nconmax)
        self.assertTrue(config.use_mujoco_contacts)
        self.assertTrue(config.include_sites)

    async def test_explicit_solver_config_takes_precedence(self) -> None:
        self._settings.set(
            self._settings_path, {"autoSize": True, "useMujocoContacts": True, "includeSites": True}
        )
        solver = MuJoCoSolverConfig(njmax=17, nconmax=11, use_mujoco_contacts=False, include_sites=False)
        configure_newton(NewtonConfig(solver_cfg=solver))
        self.assertIs(get_newton_config().solver_cfg, solver)
        self.assertEqual(get_newton_config().solver_cfg.njmax, 17)
        self.assertFalse(get_newton_config().solver_cfg.use_mujoco_contacts)

    async def test_native_contact_output_capacity(self) -> None:
        # Exercise only the integration's buffer choice, without replacing the
        # production solver or allocating large contact arrays in this unit test.
        model = SimpleNamespace(shape_flags=SimpleNamespace(numpy=lambda: np.zeros(1, dtype=np.int32)))
        for capacity, native, cpu, expected in (
            (None, True, False, 4096),
            (7, True, False, 7),
            (None, False, False, None),
            (None, True, True, None),
        ):
            with self.subTest(capacity=capacity, native=native, cpu=cpu):
                stage = object.__new__(NewtonStage)
                stage.cfg = NewtonConfig(
                    solver_cfg=MuJoCoSolverConfig(use_mujoco_contacts=native, use_mujoco_cpu=cpu),
                    collision_cfg=CollisionConfig(rigid_contact_max=capacity),
                )
                stage.solver = Mock()
                stage.solver.get_max_contact_count.return_value = 4096
                with patch("newton.CollisionPipeline") as pipeline:
                    stage._build_collision_pipeline(model)
                self.assertEqual(pipeline.call_args.kwargs["rigid_contact_max"], expected)
                if capacity is None and native and not cpu:
                    stage.solver.get_max_contact_count.assert_called_once_with()
                else:
                    stage.solver.get_max_contact_count.assert_not_called()
