#version 430 core
// fp64 is core in GLSL 4.00+, so the "#extension GL_ARB_gpu_shader_fp64 : require"
// line from the original is redundant here. Drop it if uam complains about it.
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer DataBuffer {
    double data[];
};

layout(std430, binding = 1) buffer ResultBuffer {
    uint result;
};

layout(std140, binding = 0) uniform Params {
    uint n_rows;
};

const double THRESHOLD = 1e-3LF;

void main() {
    uint total_cols = gl_NumWorkGroups.x * gl_WorkGroupSize.x
                    * gl_NumWorkGroups.y * gl_WorkGroupSize.y;

    uint col = (gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x))
             + gl_GlobalInvocationID.x;

    if (n_rows < 2u) return;

    double ref = data[col];
    uint count = 0u;

    for (uint row = 1u; row < n_rows; ++row) {
        double val = data[row * total_cols + col];
        if (abs(ref - val) > THRESHOLD)
            count++;
    }

    atomicAdd(result, count);
}
