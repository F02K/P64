N64_VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec2 in_uv;
in vec3 in_normal;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
out vec2 v_uv;
out float v_light;
out vec3 v_world_pos;
void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    vec3 normal = normalize(mat3(u_model) * in_normal);
    v_light = clamp(dot(normal, normalize(vec3(-0.4, 0.8, 0.5))) * 0.5 + 0.5, 0.18, 1.0);
    v_uv = in_uv;
    v_world_pos = world_pos.xyz;
    gl_Position = u_projection * u_view * world_pos;
}
"""

N64_FRAGMENT_SHADER = """
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
in vec2 v_uv;
in float v_light;
in vec3 v_world_pos;
out vec4 fragColor;
void main() {
    vec4 texel = texture(u_texture, v_uv);
    vec3 lit = texel.rgb * v_light;
    vec3 quantized = floor(lit * u_color_levels) / u_color_levels;
    vec3 half_size = max(u_fog_size * 0.5, vec3(0.001));
    vec3 volume_pos = abs(v_world_pos - u_fog_center) / half_size;
    float inside_volume = 1.0 - smoothstep(0.92, 1.0, max(max(volume_pos.x, volume_pos.y), volume_pos.z));
    float distance_fog = smoothstep(u_fog_near, u_fog_far, distance(v_world_pos, u_camera_position));
    float density_fog = clamp(u_fog_density, 0.0, 1.0);
    float fog_amount = u_fog_enabled ? inside_volume * max(distance_fog, density_fog) : 0.0;
    fragColor = vec4(mix(quantized, u_fog_color, fog_amount), texel.a);
}
"""
