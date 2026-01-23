import time
from ..runtime import initializer, output_manager

def run_loop(iterations=3, dt=0.016):
    cfg = initializer.initialize()
    frames = []
    for i in range(iterations):
        # minimal loop: step physics / ml / render by delegating to existing root modules if present
        # This is a lightweight orchestrator; real logic should import core modules.
        frame = f"frame_{i}"
        out = output_manager.save_frame(frame, name=f"frame_{i}.txt")
        frames.append(out)
        time.sleep(dt)
    return frames

if __name__ == "__main__":
    print(run_loop())
