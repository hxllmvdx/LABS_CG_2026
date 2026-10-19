#version 450

struct PointLight {
	vec4 position;
	vec4 color;       
};

struct SpotLight {
	vec4 position;
	vec4 direction;   
	vec4 color;       
	vec4 params;      
};

layout(set = 0, binding = 0) uniform SceneData {
	mat4 view;
	mat4 proj;
	vec4 camera_pos;
	vec4 ambient;
	vec4 sun_dir;
	vec4 sun_color;
	ivec4 counts;
} scene;

layout(set = 0, binding = 1) readonly buffer PointLightBuffer {
	PointLight lights[];
} point_lights;

layout(set = 0, binding = 2) readonly buffer SpotLightBuffer {
	SpotLight lights[];
} spot_lights;

layout(set = 1, binding = 0) uniform ObjectData {
	mat4 model;
	vec4 albedo;
	vec4 specular;
} object;

layout(location = 0) in vec3 world_pos;
layout(location = 1) in vec3 world_normal;
layout(location = 0) out vec4 out_color;

vec3 blinn_phong(vec3 n, vec3 v, vec3 l, vec3 light_rgb, float intensity,
                 vec3 albedo, vec3 spec_color, float shininess) {
	float diffuse = max(dot(n, l), 0.0);
	vec3 half_vector = normalize(l + v);
	float specular_term = pow(max(dot(n, half_vector), 0.0), shininess);
	return light_rgb * intensity * (diffuse * albedo + specular_term * spec_color);
}

void main() {
	vec3 n = normalize(world_normal);
	vec3 v = normalize(scene.camera_pos.xyz - world_pos);
	vec3 albedo = object.albedo.rgb;
	vec3 spec_color = object.specular.rgb;
	float shininess = max(object.specular.a, 1e-3);

	vec3 color = scene.ambient.rgb * scene.ambient.a * albedo;

	{
		vec3 l = normalize(scene.sun_dir.xyz);
		color += blinn_phong(n, v, l, scene.sun_color.rgb, scene.sun_color.a,
		                     albedo, spec_color, shininess);
	}

	for (int i = 0; i < scene.counts.x; ++i) {
		vec3 to_light = point_lights.lights[i].position.xyz - world_pos;
		float dist = length(to_light);
		vec3 l = to_light / max(dist, 1e-4);
		float attenuation = 1.0 / (dist * dist);
		color += blinn_phong(n, v, l, point_lights.lights[i].color.rgb,
		                     point_lights.lights[i].color.a * attenuation,
		                     albedo, spec_color, shininess);
	}

	for (int i = 0; i < scene.counts.y; ++i) {
		vec3 to_light = spot_lights.lights[i].position.xyz - world_pos;
		float dist = length(to_light);
		vec3 l = to_light / max(dist, 1e-4);
		vec3 dir = normalize(spot_lights.lights[i].direction.xyz);

		float cos_theta = dot(-l, dir);
		float cos_inner = spot_lights.lights[i].params.x;
		float cos_outer = spot_lights.lights[i].params.y;

		float spot_factor = smoothstep(cos_outer, cos_inner, cos_theta);
		float attenuation = spot_factor / (dist * dist);

		color += blinn_phong(n, v, l, spot_lights.lights[i].color.rgb,
		                     spot_lights.lights[i].color.a * attenuation,
		                     albedo, spec_color, shininess);
	}

	out_color = vec4(color, 1.0);
}
