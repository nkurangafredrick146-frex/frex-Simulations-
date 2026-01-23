"""Minimal OpenGL.GL stub providing constants and no-op functions for import-time safety."""

# Common constants
GL_VERTEX_SHADER = 0x8B31
GL_FRAGMENT_SHADER = 0x8B30
GL_GEOMETRY_SHADER = 0x8DD9
GL_FLOAT = 0x1406
GL_ARRAY_BUFFER = 0x8892
GL_DYNAMIC_DRAW = 0x88E8
GL_PROGRAM_POINT_SIZE = 0x8642
GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_ONE = 1
GL_DEPTH_TEST = 0x0B71

# Simple placeholder types
GLfloat = float

def sizeof(x):
    return 4

# No-op GL functions
def glGenVertexArrays(n):
    return 1

def glGenBuffers(n):
    return 1

def glBindBuffer(target, buffer):
    pass

def glBufferData(target, size, data, usage):
    pass

def glBindVertexArray(vao):
    pass

def glVertexAttribPointer(index, size, type, normalized, stride, pointer):
    pass

def glEnableVertexAttribArray(index):
    pass

def glDisable(cap):
    pass

def glEnable(cap):
    pass

def glUseProgram(p):
    pass

def glGetUniformLocation(program, name):
    return -1

def glUniform1i(loc, v):
    pass

def glUniform1f(loc, v):
    pass

def glUniform2f(loc, x, y):
    pass

def glUniform3f(loc, x, y, z):
    pass

def glUniform4f(loc, x, y, z, w):
    pass

def glUniformMatrix4fv(loc, count, transpose, value):
    pass

def glDrawArrays(mode, first, count):
    pass

def glDeleteVertexArrays(n, arrays):
    pass

def glDeleteBuffers(n, buffers):
    pass

def glBlendFunc(sfactor, dfactor):
    pass

def glGetError():
    return 0

__all__ = [name for name in dir() if not name.startswith('_')]
