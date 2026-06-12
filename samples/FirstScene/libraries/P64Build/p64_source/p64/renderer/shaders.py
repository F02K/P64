STANDARD_VERTEX_LIT_VERTEX_SHADER = """
#version 330
const int P64_MAX_LIGHTS = 8;

in vec3 in_position;
in vec2 in_uv;
in vec3 in_normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform int u_light_count;
uniform int u_light_kind[P64_MAX_LIGHTS];
uniform vec3 u_light_position[P64_MAX_LIGHTS];
uniform vec3 u_light_direction[P64_MAX_LIGHTS];
uniform vec3 u_light_color[P64_MAX_LIGHTS];
uniform float u_light_intensity[P64_MAX_LIGHTS];
uniform float u_light_range[P64_MAX_LIGHTS];
uniform float u_light_spot_angle[P64_MAX_LIGHTS];
uniform float u_light_falloff[P64_MAX_LIGHTS];
uniform vec3 u_ambient_color;

out vec2 v_uv;
out vec3 v_light;
out vec3 v_world_pos;

vec3 light_directional(vec3 normal, int index) {
    vec3 light_vec = normalize(-u_light_direction[index]);
    float amount = max(dot(normal, light_vec), 0.0);
    return u_light_color[index] * u_light_intensity[index] * amount;
}

vec3 light_point(vec3 normal, vec3 world_pos, int index) {
    vec3 to_light = u_light_position[index] - world_pos;
    float distance_to_light = length(to_light);
    vec3 light_vec = normalize(to_light);
    float range_value = max(u_light_range[index], 0.001);
    float attenuation = pow(clamp(1.0 - distance_to_light / range_value, 0.0, 1.0), max(u_light_falloff[index], 0.001));
    float amount = max(dot(normal, light_vec), 0.0) * attenuation;
    return u_light_color[index] * u_light_intensity[index] * amount;
}

vec3 light_spot(vec3 normal, vec3 world_pos, int index) {
    vec3 to_light = u_light_position[index] - world_pos;
    vec3 from_light = normalize(-to_light);
    vec3 spot_dir = normalize(u_light_direction[index]);
    float cutoff = cos(radians(clamp(u_light_spot_angle[index], 1.0, 179.0)) * 0.5);
    float inner = cos(radians(clamp(u_light_spot_angle[index] * 0.75, 1.0, 179.0)) * 0.5);
    float spot_amount = smoothstep(cutoff, inner, dot(from_light, spot_dir));
    return light_point(normal, world_pos, index) * spot_amount;
}

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    vec3 normal = normalize(mat3(u_model) * in_normal);
    vec3 light = u_ambient_color;
    for (int i = 0; i < min(u_light_count, P64_MAX_LIGHTS); i++) {
        if (u_light_kind[i] == 0) {
            light += light_directional(normal, i);
        } else if (u_light_kind[i] == 1) {
            light += light_point(normal, world_pos.xyz, i);
        } else if (u_light_kind[i] == 2) {
            light += light_spot(normal, world_pos.xyz, i);
        }
    }
    v_light = clamp(light, vec3(0.0), vec3(1.6));
    v_uv = in_uv;
    v_world_pos = world_pos.xyz;
    gl_Position = u_projection * u_view * world_pos;
}
"""

STANDARD_VERTEX_LIT_FRAGMENT_SHADER = """
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
in vec3 v_light;
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

STANDARD_UNLIT_VERTEX_SHADER = """
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
"""

STANDARD_UNLIT_FRAGMENT_SHADER = """
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
"""

ERROR_VERTEX_SHADER = """
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
"""

ERROR_FRAGMENT_SHADER = """
#version 330
out vec4 fragColor;
void main() {
    fragColor = vec4(1.0, 0.0, 1.0, 1.0);
}
"""
