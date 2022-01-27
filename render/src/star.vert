#version 450

layout(location = 0) out vec2 v_texcoords;
layout(location = 1) out vec3 v_color;

struct Star {
    vec3 direction;
    vec3 irradiance;
};

layout(set = 0, binding = 0) uniform Projection {
    mat4 proj;
};
layout(set = 0, binding = 1) readonly restrict buffer Stars {
    Star stars[];
};

const float RADIUS = 4e-3; // TODO: Normalize for FoV

const vec2 quad[] = vec2[](vec2(-0.5, -0.5), vec2( 0.5,  0.5), vec2( 0.5, -0.5),
                           vec2(-0.5, -0.5), vec2(-0.5,  0.5), vec2( 0.5,  0.5));

void main() {
    int vert = gl_VertexIndex % 6;
    vec2 coords = quad[vert];
    int index = gl_VertexIndex / 6;

    Star star = stars[index];
    v_color = star.irradiance;
    vec3 x = normalize(vec3(star.direction.y, -star.direction.x, 0));
    vec3 y = cross(star.direction, x);
    vec3 pos = star.direction + RADIUS * (coords.x * x + coords.y * y);
    v_texcoords = coords;
    gl_Position = proj * vec4(pos, 0);
}
