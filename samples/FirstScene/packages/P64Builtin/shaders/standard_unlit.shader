Shader "P64Builtin/Standard Unlit"
{
    Vertex
    {
        #version 330
        in vec3 in_position;
        in vec2 in_uv;
        in vec3 in_normal;
        uniform mat4 u_model;
        uniform mat4 u_view;
        uniform mat4 u_projection;
        out vec2 v_uv;
        out vec3 v_world_pos;
        void main() {
            vec4 world_pos = u_model * vec4(in_position + in_normal * 0.0, 1.0);
            v_uv = in_uv;
            v_world_pos = world_pos.xyz;
            gl_Position = u_projection * u_view * world_pos;
        }
    }

    Fragment
    {
        #version 330
        uniform sampler2D u_texture;
        uniform bool u_fog_enabled;
        uniform vec3 u_fog_color;
        uniform vec3 u_fog_center;
        uniform vec3 u_fog_size;
        uniform vec3 u_camera_position;
        uniform float u_fog_near;
        uniform float u_fog_far;
        uniform float u_fog_density;
        uniform float u_color_levels;
        uniform int u_texture_filter;
        in vec2 v_uv;
        in vec3 v_world_pos;
        out vec4 fragColor;

        vec4 sample_three_point(sampler2D tex, vec2 uv) {
            vec2 size = vec2(textureSize(tex, 0));
            vec2 texel = uv * size - vec2(0.5);
            vec2 base = floor(texel);
            vec2 fraction = fract(texel);
            vec2 uv00 = (base + vec2(0.5, 0.5)) / size;
            vec2 uv10 = (base + vec2(1.5, 0.5)) / size;
            vec2 uv01 = (base + vec2(0.5, 1.5)) / size;
            vec4 c00 = texture(tex, uv00);
            vec4 c10 = texture(tex, uv10);
            vec4 c01 = texture(tex, uv01);
            return c00 + (c10 - c00) * fraction.x + (c01 - c00) * fraction.y;
        }

        void main() {
            vec4 texel = u_texture_filter == 2 ? sample_three_point(u_texture, v_uv) : texture(u_texture, v_uv);
            vec3 quantized = floor(texel.rgb * u_color_levels) / u_color_levels;
            vec3 half_size = max(u_fog_size * 0.5, vec3(0.001));
            vec3 volume_pos = abs(v_world_pos - u_fog_center) / half_size;
            float inside_volume = 1.0 - smoothstep(0.92, 1.0, max(max(volume_pos.x, volume_pos.y), volume_pos.z));
            float distance_fog = smoothstep(u_fog_near, u_fog_far, distance(v_world_pos, u_camera_position));
            float density_fog = clamp(u_fog_density, 0.0, 1.0);
            float fog_amount = u_fog_enabled ? inside_volume * max(distance_fog, density_fog) : 0.0;
            fragColor = vec4(mix(quantized, u_fog_color, fog_amount), texel.a);
        }
    }
}
