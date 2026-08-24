"""Fixed-root ShadowHand18 task with object-only gravity variation."""

from isaacgymenvs.tasks.object_gravity_schedule import ObjectGravityScheduleMixin
from isaacgymenvs.tasks.shadow_hand import ShadowHand


class ShadowHandGravity(ObjectGravityScheduleMixin, ShadowHand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure_object_gravity_schedule(self.cfg)

    def post_physics_step(self):
        super().post_physics_step()
        self._publish_gravity_schedule_metrics()
