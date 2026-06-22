Shader "P64Builtin/Skybox"
{
    Properties
    {
        Color u_skybox_top_color = (0.22, 0.48, 0.86)
        Color u_skybox_horizon_color = (0.66, 0.82, 0.95)
        Float u_color_levels = 32.0 Range(2, 256)
        Bool u_dithering_enabled = true
    }

    Vertex
    {
        #version 330
        in vec2 in_position;
        out vec2 v_uv;
        void main() {
            v_uv = in_position * 0.5 + 0.5;
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
    }

    Fragment
    {
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
    }
}
