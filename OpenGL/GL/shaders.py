"""Minimal shader helpers stub for import-time safety."""

def compileShader(source, shader_type):
    # Return a dummy shader id
    return 1

def compileProgram(*shaders):
    # Return a dummy program id
    return 1

__all__ = ["compileShader", "compileProgram"]
