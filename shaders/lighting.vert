#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

// Set 0: данные сцены (общие для всех объектов)
layout(set = 0, binding = 0) uniform SceneData {
	mat4 view;
	mat4 proj;
	vec4 camera_pos;   // xyz + не используется
	vec4 ambient;      // rgb + интенсивность
	vec4 sun_dir;      // направление К солнцу (xyz)
	vec4 sun_color;    // rgb + интенсивность
	ivec4 counts;      // x - точечные источники, y - прожекторы
} scene;

// Set 1: данные модели (свои для каждого объекта)
layout(set = 1, binding = 0) uniform ObjectData {
	mat4 model;
	vec4 albedo;       // rgb = альбедо
	vec4 specular;     // rgb = цвет блика, a = коэффициент блика
} object;

layout(location = 0) out vec3 world_pos;
layout(location = 1) out vec3 world_normal;

void main() {
	vec4 wp = object.model * vec4(in_position, 1.0);
	world_pos = wp.xyz;
	world_normal = mat3(object.model) * in_normal;
	gl_Position = scene.proj * scene.view * wp;
}
