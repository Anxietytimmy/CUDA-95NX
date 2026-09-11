#version 430 core
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer DataBuffer {
    float data[];
};

layout(std430, binding = 1) buffer ResultBuffer {
    uint result;
};

// uam rejects default-block uniforms ("uniform uint n_rows;") because DKSH has
// no way to report a uniform location. Everything must live in an explicitly
// bound UBO instead. Compute UBO bindings 0-5 are native; 6-15 are emulated.
layout(std140, binding = 0) uniform Params {
    uint n_rows;
};

const float THRESHOLD = 0.001f;

void main() {
    uint total_cols = gl_NumWorkGroups.x * gl_WorkGroupSize.x
                    * gl_NumWorkGroups.y * gl_WorkGroupSize.y;

    uint col = (gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x))
             + gl_GlobalInvocationID.x;

    if (n_rows < 2u) return;

    float ref = data[col];
    uint count = 0u;

    for (uint row = 1u; row < n_rows; ++row) {
        float val = data[row * total_cols + col];
        if (abs(ref - val) > THRESHOLD)
            count++;
    }

    atomicAdd(result, count);
}
