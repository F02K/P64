STANDARD_VERTEX_LIT_VERTEX_SHADER = """
#version 330
const int P64_MAX_LIGHTS = 8;

in vec3 in_position;
in vec2 in_uv;
in vec3 in_normal;
in vec3 in_color;

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
out vec3 v_color;
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
    v_color = in_color;
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
uniform bool u_dithering_enabled;
uniform vec3 u_base_color;
in vec2 v_uv;
in vec3 v_color;
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

float dither_threshold(vec2 position) {
    int x = int(mod(position.x, 4.0));
    int y = int(mod(position.y, 4.0));
    int index = x + y * 4;
    float values[16] = float[16](
        0.0, 8.0, 2.0, 10.0,
        12.0, 4.0, 14.0, 6.0,
        3.0, 11.0, 1.0, 9.0,
        15.0, 7.0, 13.0, 5.0
    );
    return (values[index] / 16.0) - 0.5;
}

vec3 quantize_color(vec3 color) {
    float levels = max(u_color_levels, 2.0);
    vec3 adjusted = color;
    if (u_dithering_enabled) {
        adjusted += vec3(dither_threshold(gl_FragCoord.xy) / levels);
    }
    return floor(clamp(adjusted, vec3(0.0), vec3(1.0)) * levels) / levels;
}

void main() {
    vec4 texel = u_texture_filter == 2 ? sample_three_point(u_texture, v_uv) : texture(u_texture, v_uv);
    vec3 lit = texel.rgb * u_base_color * v_color * v_light;
    vec3 quantized = quantize_color(lit);
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
in vec3 in_color;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
out vec2 v_uv;
out vec3 v_color;
out vec3 v_world_pos;
void main() {
    vec4 world_pos = u_model * vec4(in_position + in_normal * 0.0, 1.0);
    v_uv = in_uv;
    v_color = in_color;
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
uniform bool u_dithering_enabled;
uniform vec3 u_base_color;
in vec2 v_uv;
in vec3 v_color;
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

float dither_threshold(vec2 position) {
    int x = int(mod(position.x, 4.0));
    int y = int(mod(position.y, 4.0));
    int index = x + y * 4;
    float values[16] = float[16](
        0.0, 8.0, 2.0, 10.0,
        12.0, 4.0, 14.0, 6.0,
        3.0, 11.0, 1.0, 9.0,
        15.0, 7.0, 13.0, 5.0
    );
    return (values[index] / 16.0) - 0.5;
}

vec3 quantize_color(vec3 color) {
    float levels = max(u_color_levels, 2.0);
    vec3 adjusted = color;
    if (u_dithering_enabled) {
        adjusted += vec3(dither_threshold(gl_FragCoord.xy) / levels);
    }
    return floor(clamp(adjusted, vec3(0.0), vec3(1.0)) * levels) / levels;
}

void main() {
    vec4 texel = u_texture_filter == 2 ? sample_three_point(u_texture, v_uv) : texture(u_texture, v_uv);
    vec3 quantized = quantize_color(texel.rgb * u_base_color * v_color);
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

SKYBOX_VERTEX_SHADER = """
#version 330
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

SKYBOX_FRAGMENT_SHADER = """
#version 330
uniform vec3 u_skybox_top_color;
uniform vec3 u_skybox_horizon_color;
uniform float u_color_levels;
uniform bool u_dithering_enabled;
in vec2 v_uv;
out vec4 fragColor;

float dither_threshold(vec2 position) {
    int x = int(mod(position.x, 4.0));
    int y = int(mod(position.y, 4.0));
    int index = x + y * 4;
    float values[16] = float[16](
        0.0, 8.0, 2.0, 10.0,
        12.0, 4.0, 14.0, 6.0,
        3.0, 11.0, 1.0, 9.0,
        15.0, 7.0, 13.0, 5.0
    );
    return (values[index] / 16.0) - 0.5;
}

vec3 quantize_color(vec3 color) {
    float levels = max(u_color_levels, 2.0);
    vec3 adjusted = color;
    if (u_dithering_enabled) {
        adjusted += vec3(dither_threshold(gl_FragCoord.xy) / levels);
    }
    return floor(clamp(adjusted, vec3(0.0), vec3(1.0)) * levels) / levels;
}

void main() {
    float height = smoothstep(0.0, 1.0, v_uv.y);
    vec3 color = mix(u_skybox_horizon_color, u_skybox_top_color, height);
    fragColor = vec4(quantize_color(color), 1.0);
}
"""

CLOUD_PLANE_VERTEX_SHADER = """
#version 330
in vec3 in_position;
uniform mat4 u_view;
uniform mat4 u_projection;
out vec2 v_world_xz;
void main() {
    v_world_xz = in_position.xz;
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
}
"""

CLOUD_PLANE_FRAGMENT_SHADER = """
#version 330
uniform vec3 u_skybox_cloud_color;
uniform float u_skybox_cloud_coverage;
uniform float u_skybox_cloud_scale;
uniform float u_skybox_cloud_height;
uniform float u_skybox_cloud_softness;
uniform float u_color_levels;
uniform bool u_dithering_enabled;
in vec2 v_world_xz;
out vec4 fragColor;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 345.45));
    p += dot(p, p + 34.345);
    return fract(p.x * p.y);
}

float value_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    return value_noise(p) * 0.58 + value_noise(p * 2.0 + 17.0) * 0.28 + value_noise(p * 4.0 + 43.0) * 0.14;
}

float dither_threshold(vec2 position) {
    int x = int(mod(position.x, 4.0));
    int y = int(mod(position.y, 4.0));
    int index = x + y * 4;
    float values[16] = float[16](
        0.0, 8.0, 2.0, 10.0,
        12.0, 4.0, 14.0, 6.0,
        3.0, 11.0, 1.0, 9.0,
        15.0, 7.0, 13.0, 5.0
    );
    return (values[index] / 16.0) - 0.5;
}

vec3 quantize_color(vec3 color) {
    float levels = max(u_color_levels, 2.0);
    vec3 adjusted = color;
    if (u_dithering_enabled) {
        adjusted += vec3(dither_threshold(gl_FragCoord.xy) / levels);
    }
    return floor(clamp(adjusted, vec3(0.0), vec3(1.0)) * levels) / levels;
}

void main() {
    vec2 uv = v_world_xz / max(u_skybox_cloud_height, 1.0) * max(u_skybox_cloud_scale, 0.1);
    float noise = fbm(uv);
    float threshold = mix(0.86, 0.34, clamp(u_skybox_cloud_coverage, 0.0, 1.0));
    float softness = max(u_skybox_cloud_softness, 0.001);
    float cloud = smoothstep(threshold, threshold + softness, noise);
    cloud = floor(cloud * 4.0) / 4.0;
    if (cloud <= 0.001) {
        discard;
    }
    fragColor = vec4(quantize_color(u_skybox_cloud_color), cloud * 0.9);
}
"""
