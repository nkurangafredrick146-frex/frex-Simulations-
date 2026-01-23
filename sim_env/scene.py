# sim_env/scene.py

class Scene:
    def __init__(self, name="DefaultScene"):
        self.name = name

    def render(self):
        return f"Rendering scene: {self.name}"