#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

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
layout(location = 2) out vec2 uv;

void main() {
	uv = in_uv;
	if (object.specular.a < 0.0) {
		vec2 fullscreen_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
		world_pos = vec3(fullscreen_uv, 0.0);
		world_normal = vec3(0.0, 0.0, 1.0);
		gl_Position = vec4(fullscreen_uv * 2.0 - 1.0, 0.0, 1.0);
		return;
	}

	vec4 wp = object.model * vec4(in_position, 1.0);
	world_pos = wp.xyz;
	world_normal = mat3(object.model) * in_normal;
	gl_Position = scene.proj * scene.view * wp;
}
