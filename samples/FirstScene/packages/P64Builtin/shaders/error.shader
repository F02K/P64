Shader "P64Builtin/Error"
{
    Properties
    {

    }

    Vertex
    {
        #version 330
        in vec3 in_position;
        in vec2 in_uv;
        in vec3 in_normal;
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;
        void main() {
            vec3 position = in_position + (in_normal * 0.0) + vec3(in_uv, 0.0) * 0.0;
            gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
        }
    }

    Fragment
    {
        #version 330
        out vec4 fragColor;
        void main() {
            fragColor = vec4(1.0, 0.0, 1.0, 1.0);
        }
    }
}
