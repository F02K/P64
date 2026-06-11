from p64.engine.scripting import UserScript


class Spin(UserScript):
    speed = 45.0

    def on_update(self, dt):
        self.transform.rotation.y += self.speed * dt
