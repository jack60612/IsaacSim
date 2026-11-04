# SPDX-FileCopyrightText: Copyright (c) 2026 Jack
# SPDX-License-Identifier: Apache-2.0

"""Optional viewport gesture bridge to Newton's Python raycast callback.

Imported only while ``omni.kit.viewport.window`` is enabled; the Newton backend
must also work without any viewport or UI extensions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import carb.input
import omni.kit.app
import omni.ui as ui
from omni.kit.viewport.window import get_viewport_window_instances
from omni.timeline import get_timeline_interface
from omni.ui_scene import scene as sc


class _PassThroughGestures(sc.GestureManager):
    """Observe dragging without preventing the viewport's other gestures."""

    def can_be_prevented(self, gesture: Any) -> bool:
        return False

    def should_prevent(self, gesture: Any, preventer: Any) -> bool:
        return False


class _ViewportOverlay:
    def __init__(self, window: Any, owner: NewtonViewportInteraction) -> None:
        # Enumeration returns weak proxies. Do not keep a viewport window alive.
        self.window = window
        self._owner = owner
        self._ray = None
        self._dragging = False
        self._input = carb.input.acquire_input_interface()
        self._keyboard = window.app_window.get_keyboard()
        self._frame = window.get_frame("isaacsim.physics.newton.interaction")
        with self._frame:
            self._container = ui.VStack(content_clipping=False)
            with self._container:
                self._scene_view = sc.SceneView()
                window.viewport_api.add_scene_view(self._scene_view)
                with self._scene_view.scene:
                    sc.Screen(
                        gesture=[
                            sc.DragGesture(
                                mouse_button=0,
                                manager=_PassThroughGestures(),
                                on_began_fn=self._on_began,
                                on_changed_fn=self._on_changed,
                                on_ended_fn=self._on_ended,
                            )
                        ]
                    )
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)
        self.refresh()

    def refresh(self) -> None:
        enabled = self._owner.is_enabled()
        if not enabled:
            self.release()
        shift = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.LEFT_SHIFT) or (
            self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.RIGHT_SHIFT)
        )
        # Match the stock physics overlay: intercept picking only during Shift
        # interaction, and keep receiving the end even if Shift is released first.
        self._container.content_clipping = enabled and (bool(shift) or self._dragging)

    def release(self) -> None:
        if self._dragging:
            self._dragging = False
            # Use the last valid ray, including when the window has disappeared.
            self._owner.handle_raycast(*self._ray, False)
            if self._owner.active_overlay is self:
                self._owner.active_overlay = None

    def destroy(self) -> None:
        self.release()
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None
        try:
            api = self.window.viewport_api
            if api:
                api.remove_scene_view(self._scene_view)
        except ReferenceError:
            pass
        self._scene_view.scene.clear()
        self._frame.clear()
        self._scene_view = None
        self._frame = None

    def _on_key(self, event: Any) -> bool:
        self.refresh()
        return True

    def _send_ray(self, sender: Any) -> None:
        origin, direction = self._scene_view.get_ray_from_ndc(sc.Vector2(*sender.gesture_payload.mouse))
        self._ray = (
            carb.Float3(origin[0], origin[1], origin[2]),
            carb.Float3(direction[0], direction[1], direction[2]),
        )
        self._owner.handle_raycast(*self._ray, True)

    def _on_began(self, sender: Any) -> None:
        if not self._owner.is_enabled():
            return
        if not (
            self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.LEFT_SHIFT)
            or self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.RIGHT_SHIFT)
        ):
            return
        if self._owner.active_overlay is not None:
            self._owner.active_overlay.release()
        self._owner.active_overlay = self
        self._dragging = True
        self._send_ray(sender)
        self.refresh()

    def _on_changed(self, sender: Any) -> None:
        if self._dragging:
            if self._owner.is_enabled():
                self._send_ray(sender)
            else:
                self.release()

    def _on_ended(self, sender: Any) -> None:
        self.release()
        self.refresh()


class NewtonViewportInteraction:
    """Maintain Shift-left-drag overlays for the default USD context's windows.

    Args:
        is_newton_active: Query whether Newton is the active physics backend.
        handle_raycast: Python callback preserving all three ray components.
    """

    def __init__(
        self,
        is_newton_active: Callable[[], bool],
        handle_raycast: Callable[[carb.Float3, carb.Float3, bool], bool],
    ) -> None:
        self.is_newton_active = is_newton_active
        # Native callback bindings in supported Kit builds declare scalar ray
        # arguments. Keep vectors intact by calling the Python adapter directly.
        self.handle_raycast = handle_raycast
        self._timeline = get_timeline_interface()
        self._overlays: dict[int, _ViewportOverlay] = {}
        self.active_overlay: _ViewportOverlay | None = None
        # ViewportWindow exposes enumeration, but no window-created/destroyed
        # subscription. Reconcile weak window proxies on Kit updates instead.
        self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name="Newton viewport interaction"
        )
        self._on_update(None)

    def is_enabled(self) -> bool:
        """Return whether viewport rays may be sent to Newton."""
        return self._timeline.is_playing() and self.is_newton_active()

    def destroy(self) -> None:
        """Release interaction, subscriptions, and all viewport scene views."""
        self._update_sub = None
        for overlay in self._overlays.values():
            overlay.destroy()
        self._overlays.clear()

    def _on_update(self, event: Any) -> None:
        windows = {id(window.viewport_api): window for window in get_viewport_window_instances()}
        for key in tuple(self._overlays):
            if key not in windows:
                self._overlays.pop(key).destroy()
        for key, window in windows.items():
            overlay = self._overlays.get(key)
            if overlay is None:
                self._overlays[key] = _ViewportOverlay(window, self)
            else:
                overlay.refresh()
