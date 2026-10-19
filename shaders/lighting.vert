#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(set = 0, binding = 0) uniform SceneData {
	mat4 view;
	mat4 proj;
	vec4 camera_pos;  
	vec4 ambient;     
	vec4 sun_dir;     
	vec4 sun_color;   
	ivec4 counts;     
} scene;

layout(set = 1, binding = 0) uniform ObjectData {
	mat4 model;
	vec4 albedo;      
	vec4 specular;    
} object;

layout(location = 0) out vec3 world_pos;
layout(location = 1) out vec3 world_normal;

void main() {
	vec4 wp = object.model * vec4(in_position, 1.0);
	world_pos = wp.xyz;
	world_normal = mat3(object.model) * in_normal;
	gl_Position = scene.proj * scene.view * wp;
}
