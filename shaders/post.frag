#version 450

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

layout(set = 2, binding = 0) uniform sampler2D scene_image;
layout(location = 0) out vec4 out_color;

vec3 bright_sample(vec2 uv, float threshold) {
	vec3 c = texture(scene_image, uv).rgb;
	float brightness = max(c.r, max(c.g, c.b));
	return c * max(brightness - threshold, 0.0) / max(brightness, 0.001);
}

void main() {
	vec2 resolution = vec2(textureSize(scene_image, 0));
	vec2 uv = gl_FragCoord.xy / resolution;
	vec2 texel = 1.0 / resolution;
	vec3 color = texture(scene_image, uv).rgb;

	if ((scene.counts.z & 2) != 0) {
		float threshold = object.model[1].x;
		float intensity = object.model[1].y;
		vec3 bloom = vec3(0.0);
		vec2 offsets[12] = vec2[](
			vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0),
			vec2(0.7, 0.7), vec2(-0.7, 0.7), vec2(0.7, -0.7), vec2(-0.7, -0.7),
			vec2(2.0, 0.0), vec2(-2.0, 0.0), vec2(0.0, 2.0), vec2(0.0, -2.0));
		for (int i = 0; i < 12; ++i)
			bloom += bright_sample(uv + offsets[i] * texel * object.model[1].z, threshold);
		color += bloom * (intensity / 12.0);
	}

	color = color / (color + vec3(1.0));
	color = pow(max(color, 0.0), vec3(1.0 / 2.2));
	out_color = vec4(color, 1.0);
}
