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
	vec4 effect;       // x = включено, y = амплитуда, z = частота, w = скорость
} object;

layout(set = 1, binding = 1) uniform sampler2D surface_texture;

layout(location = 0) in vec3 world_pos;
layout(location = 1) in vec3 world_normal;
layout(location = 2) in vec2 uv;
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
	vec2 texture_uv = vec2(uv.x, 1.0 - uv.y) * object.albedo.a;
	vec3 texel;
	if (object.effect.x > 0.5) {
		float time = float(scene.counts.w) * 0.001 * object.effect.w;
		vec2 wave = vec2(
			sin(texture_uv.y * object.effect.z + time),
			cos(texture_uv.x * object.effect.z * 0.83 + time * 1.17));
		wave += 0.45 * vec2(
			sin(texture_uv.y * object.effect.z * 1.9 - time * 0.7),
			cos(texture_uv.x * object.effect.z * 1.6 + time * 0.9));
		vec2 warped_uv = texture_uv + wave * object.effect.y;
		float dispersion = object.effect.y * 0.12;
		texel.r = texture(surface_texture, warped_uv + vec2(dispersion, 0.0)).r;
		texel.g = texture(surface_texture, warped_uv).g;
		texel.b = texture(surface_texture, warped_uv - vec2(dispersion, 0.0)).b;
	} else {
		texel = texture(surface_texture, texture_uv).rgb;
	}
	vec3 albedo = object.albedo.rgb * texel;
	vec3 spec_color = object.specular.rgb;
	float shininess = max(object.specular.a, 1e-3);

	vec3 color = scene.ambient.rgb * scene.ambient.a * albedo;
	bool toon_enabled = (scene.counts.z & 1) != 0;

	{
		vec3 l = normalize(scene.sun_dir.xyz);
		float diffuse = max(dot(n, l), 0.0);
		if (toon_enabled) {
			float bands = diffuse > 0.65 ? 1.0 : (diffuse > 0.25 ? 0.55 : 0.18);
			color += scene.sun_color.rgb * scene.sun_color.a * albedo * bands;
			float highlight = step(0.985, dot(n, normalize(l + v)));
			color += scene.sun_color.rgb * scene.sun_color.a * spec_color * highlight * 0.35;
		} else {
			color += blinn_phong(n, v, l, scene.sun_color.rgb, scene.sun_color.a,
			                     albedo, spec_color, shininess);
		}
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

	if (toon_enabled) {
		float rim = 1.0 - max(dot(n, v), 0.0);
		float outline = smoothstep(0.68, 0.82, rim);
		color = mix(color, albedo * 0.035, outline);
	}

	out_color = vec4(color, 1.0);
}
