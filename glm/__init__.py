"""Minimal glm stub for local import-time compatibility.
This provides small Vec/Mat classes and utility functions used by the project.
Not a full replacement for the real glm library.
"""
import math
import numpy as np

class vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        if isinstance(x, (list, tuple, np.ndarray)):
            self.x, self.y, self.z = float(x[0]), float(x[1]), float(x[2])
        else:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, s):
        return vec3(self.x * s, self.y * s, self.z * s)
    def __rmul__(self, s):
        return self.__mul__(s)
    def __iadd__(self, other):
        self.x += other.x; self.y += other.y; self.z += other.z; return self
    def __repr__(self):
        return f"vec3({self.x}, {self.y}, {self.z})"

class mat4(np.ndarray):
    @staticmethod
    def identity():
        return np.eye(4, dtype=np.float32).view(mat4)

def mat4_identity():
    return mat4.identity()

def value_ptr(m):
    # Return a flat array suitable for C interop — many callers only need an object
    return np.asarray(m).astype(np.float32).ravel()

def normalize(v):
    a = np.array([v.x, v.y, v.z], dtype=np.float32)
    n = np.linalg.norm(a)
    if n == 0:
        return vec3(a[0], a[1], a[2])
    a /= n
    return vec3(a[0], a[1], a[2])

def lookAt(eye, center, up):
    # Simple lookAt producing a 4x4 view matrix (numpy array)
    eye_a = np.array([eye.x, eye.y, eye.z], dtype=np.float32)
    center_a = np.array([center.x, center.y, center.z], dtype=np.float32)
    up_a = np.array([up.x, up.y, up.z], dtype=np.float32)
    f = center_a - eye_a
    f = f / (np.linalg.norm(f) + 1e-12)
    u = up_a / (np.linalg.norm(up_a) + 1e-12)
    s = np.cross(f, u)
    s = s / (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye_a
    return (M @ T).view(mat4)

def perspective(fovy, aspect, znear, zfar):
    f = 1.0 / math.tan(fovy / 2.0)
    M = np.zeros((4,4), dtype=np.float32)
    M[0,0] = f / aspect
    M[1,1] = f
    M[2,2] = (zfar + znear) / (znear - zfar)
    M[2,3] = (2 * zfar * znear) / (znear - zfar)
    M[3,2] = -1.0
    return M.view(mat4)

def radians(deg):
    return math.radians(deg)

def distance(a, b):
    ax = np.array([a.x, a.y, a.z], dtype=np.float32)
    bx = np.array([b.x, b.y, b.z], dtype=np.float32)
    return float(np.linalg.norm(ax - bx))

__all__ = ["vec3", "mat4", "mat4_identity", "value_ptr", "normalize", "lookAt", "perspective", "radians", "distance"]
