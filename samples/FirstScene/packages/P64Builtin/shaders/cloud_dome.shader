Shader "P64Builtin/Cloud Dome"
{
    Properties
    {
        Color u_skybox_cloud_color = (1.0, 0.96, 0.86)
        Float u_skybox_cloud_coverage = 0.45 Range(0, 1)
        Float u_skybox_cloud_scale = 3.0 Range(0.1, 24)
        Float u_skybox_cloud_height = 80.0 Range(0.1, 10000)
        Float u_skybox_cloud_softness = 0.08 Range(0, 1)
        Float u_color_levels = 32.0 Range(2, 256)
        Bool u_dithering_enabled = true
    }

    Vertex
    {
        #version 330
        in vec3 in_position;
        uniform mat4 u_view;
        uniform mat4 u_projection;
        uniform vec3 u_cloud_origin;
        uniform float u_skybox_cloud_height;
        out vec2 v_cloud_uv;
        out float v_dome_height;
        void main() {
            vec3 world_position = in_position + u_cloud_origin;
            v_cloud_uv = in_position.xz / max(u_skybox_cloud_height, 1.0);
            v_dome_height = clamp(in_position.y / max(u_skybox_cloud_height, 1.0), 0.0, 1.0);
            gl_Position = u_projection * u_view * vec4(world_position, 1.0);
        }
    }

    Fragment
    {
        #version 330
        uniform vec3 u_skybox_cloud_color;
        uniform float u_skybox_cloud_coverage;
        uniform float u_skybox_cloud_scale;
        uniform float u_skybox_cloud_height;
        uniform float u_skybox_cloud_softness;
        uniform float u_color_levels;
        uniform bool u_dithering_enabled;
        in vec2 v_cloud_uv;
        in float v_dome_height;
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
            vec2 uv = v_cloud_uv * max(u_skybox_cloud_scale, 0.1);
            float noise = fbm(uv);
            float threshold = mix(0.86, 0.34, clamp(u_skybox_cloud_coverage, 0.0, 1.0));
            float softness = max(u_skybox_cloud_softness, 0.001);
            float cloud = smoothstep(threshold, threshold + softness, noise);
            float horizon_fade = smoothstep(0.08, 0.30, v_dome_height);
            float zenith_fade = 1.0 - smoothstep(0.88, 1.0, v_dome_height);
            cloud *= horizon_fade * zenith_fade;
            cloud = floor(cloud * 4.0) / 4.0;
            if (cloud <= 0.001) {
                discard;
            }
            fragColor = vec4(quantize_color(u_skybox_cloud_color), cloud * 0.9);
        }
    }
}
