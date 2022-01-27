#version 450

layout(location = 0) in vec2 v_texcoords;
layout(location = 1) flat in vec3 v_color;

layout(location = 0) out vec4 f_color;

const float SCALE = 24;

void main() {
    float blur = exp(-SCALE * dot(v_texcoords, v_texcoords));
    f_color = vec4(blur * v_color, blur);
}
