#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(set = 0, binding = 0) uniform GlobalTransforms {
	mat4 view;
	mat4 proj;
} global_transforms;

layout(set = 1, binding = 0) uniform ObjectTransforms {
	mat4 model;
	vec4 tint;
} object_transforms;

layout(location = 0) out vec3 vertex_color;

void main() {
	gl_Position = global_transforms.proj * global_transforms.view
	             * object_transforms.model * vec4(in_position, 1.0);
	vertex_color = in_color * object_transforms.tint.rgb;
}
