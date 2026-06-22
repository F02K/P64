Shader "P64Builtin/Sprite"
{
    Properties
    {
        Texture u_texture = ""
        Color u_base_color = (1.0, 1.0, 1.0)
        Float u_alpha = 1.0 Range(0, 1)
        Float u_alpha_cutoff = 0.0 Range(0, 1)
    }

    Vertex
    {
        #version 330
        in vec3 in_position;
        in vec2 in_uv;
        in vec3 in_color;
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;
        out vec2 v_uv;
        out vec3 v_color;
        void main() {
            v_uv = in_uv;
            v_color = in_color;
            gl_Position = u_projection * u_view * u_model * vec4(in_position, 1.0);
        }
    }

    Fragment
    {
        #version 330
        uniform sampler2D u_texture;
        uniform vec3 u_base_color;
        uniform float u_alpha;
        uniform float u_alpha_cutoff;
        in vec2 v_uv;
        in vec3 v_color;
        out vec4 fragColor;
        void main() {
            vec4 texel = texture(u_texture, v_uv);
            float alpha = texel.a * clamp(u_alpha, 0.0, 1.0);
            if (alpha < u_alpha_cutoff) {
                discard;
            }
            fragColor = vec4(texel.rgb * u_base_color * v_color, alpha);
        }
    }
}
