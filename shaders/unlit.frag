#version 450

// Постоянный цвет, без освещения - для визуализации источников света

layout(set = 1, binding = 0) uniform ObjectData {
	mat4 model;
	vec4 albedo;
	vec4 specular;
} object;

layout(location = 0) out vec4 out_color;

void main() {
	out_color = vec4(object.albedo.rgb, 1.0);
}
