#include "application.hpp"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace application {

namespace {

struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
};

struct SceneData {
  glm::mat4 view;
  glm::mat4 proj;
  alignas(16) glm::vec4 camera_pos;
  alignas(16) glm::vec4 ambient;
  alignas(16) glm::vec4 sun_dir;
  alignas(16) glm::vec4 sun_color;
  alignas(16) glm::ivec4 counts;
};

struct ObjectData {
  glm::mat4 model;
  alignas(16) glm::vec4 albedo;
  alignas(16) glm::vec4 specular;
};

struct GpuPointLight {
  alignas(16) glm::vec4 position;
  alignas(16) glm::vec4 color;
};

struct GpuSpotLight {
  alignas(16) glm::vec4 position;
  alignas(16) glm::vec4 direction;
  alignas(16) glm::vec4 color;
  alignas(16) glm::vec4 params;
};

struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
};

struct GpuObject {
  Mesh mesh;

  VkBuffer vertex_buffer = VK_NULL_HANDLE;
  VmaAllocation vertex_allocation = nullptr;
  VkBuffer index_buffer = VK_NULL_HANDLE;
  VmaAllocation index_allocation = nullptr;

  VkBuffer uniform_buffer = VK_NULL_HANDLE;
  VmaAllocation uniform_allocation = nullptr;
  void *uniform_mapped = nullptr;

  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

Mesh make_box_mesh(glm::vec3 min_corner, glm::vec3 max_corner) {
  const glm::vec3 p[8] = {
      {min_corner.x, min_corner.y, min_corner.z},
      {max_corner.x, min_corner.y, min_corner.z},
      {max_corner.x, max_corner.y, min_corner.z},
      {min_corner.x, max_corner.y, min_corner.z},
      {min_corner.x, min_corner.y, max_corner.z},
      {max_corner.x, min_corner.y, max_corner.z},
      {max_corner.x, max_corner.y, max_corner.z},
      {min_corner.x, max_corner.y, max_corner.z},
  };

  Mesh mesh;

  auto face = [&](int a, int b, int c, int d, glm::vec3 normal) {
    const uint32_t base = uint32_t(mesh.vertices.size());

    for (int i : {a, b, c, d}) {
      mesh.vertices.push_back({p[i], normal});
    }

    mesh.indices.insert(mesh.indices.end(), {
                                                base + 0,
                                                base + 1,
                                                base + 2,
                                                base + 0,
                                                base + 2,
                                                base + 3,
                                            });
  };

  face(4, 5, 6, 7, {0.0f, 0.0f, 1.0f});  // +Z
  face(1, 0, 3, 2, {0.0f, 0.0f, -1.0f}); // -Z
  face(5, 1, 2, 6, {1.0f, 0.0f, 0.0f});  // +X
  face(0, 4, 7, 3, {-1.0f, 0.0f, 0.0f}); // -X
  face(7, 6, 2, 3, {0.0f, 1.0f, 0.0f});  // +Y
  face(0, 1, 5, 4, {0.0f, -1.0f, 0.0f}); // -Y

  return mesh;
}

constexpr int MAX_LIGHTS = 16;

struct CameraState {
  glm::vec3 position{0.0f, 1.3f, 3.4f};
  float yaw = -90.0f;
  float pitch = -12.0f;
  float fov = 60.0f;
};

struct PointLightState {
  glm::vec3 position{0.0f, 1.8f, 0.4f};
  glm::vec3 color{1.0f, 0.85f, 0.6f};
  float intensity = 4.0f;
};

struct SpotLightState {
  glm::vec3 position{0.0f, 2.4f, 0.0f};
  glm::vec3 direction{0.0f, -1.0f, 0.0f};
  glm::vec3 color{0.7f, 0.8f, 1.0f};
  float intensity = 6.0f;
  float inner_angle = 20.0f;
  float outer_angle = 32.0f;
};

struct MaterialState {
  glm::vec3 albedo{0.75f, 0.72f, 0.68f};
  glm::vec3 specular{1.0f, 1.0f, 1.0f};
  float shininess = 32.0f;
};

struct AppState {
  CameraState camera;

  glm::vec3 ambient_color{1.0f, 1.0f, 1.0f};
  float ambient_intensity = 0.06f;

  glm::vec3 sun_direction{0.45f, 1.0f, 0.3f};
  glm::vec3 sun_color{1.0f, 0.96f, 0.88f};
  float sun_intensity = 0.7f;

  std::vector<PointLightState> point_lights{PointLightState{}};
  std::vector<SpotLightState> spot_lights{SpotLightState{}};

  MaterialState material;

  bool show_light_gizmos = true;
};

AppState state;

VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline_lit;
VkPipeline vk_pipeline_unlit;
VkPipeline vk_pipeline_lines;

VkDescriptorSetLayout vk_set_layouts[2];
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_scene_descriptor_set;

VkBuffer vk_scene_uniform_buffer;
VmaAllocation vk_scene_uniform_allocation;
void *vk_scene_uniform_mapped = nullptr;

VkBuffer vk_point_lights_buffer;
VmaAllocation vk_point_lights_allocation;
void *vk_point_lights_mapped = nullptr;

VkBuffer vk_spot_lights_buffer;
VmaAllocation vk_spot_lights_allocation;
void *vk_spot_lights_mapped = nullptr;

std::vector<GpuObject> objects;

GpuObject gizmo_cube;
constexpr uint32_t GIZMO_SLOT_COUNT = MAX_LIGHTS * 2 + 1;
constexpr VkDeviceSize GIZMO_SLOT_STRIDE = 256;
VkBuffer vk_gizmo_uniform_buffer;
VmaAllocation vk_gizmo_uniform_allocation;
void *vk_gizmo_uniform_mapped = nullptr;
VkDescriptorSet vk_gizmo_descriptor_sets[GIZMO_SLOT_COUNT];

VkBuffer vk_lines_vertex_buffer;
VmaAllocation vk_lines_vertex_allocation;
void *vk_lines_vertex_mapped = nullptr;
uint32_t vk_lines_vertex_capacity = 0;
uint32_t vk_lines_vertex_count = 0;

std::vector<char> read_shader_file(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Failed to open shader file: " << path << '\n';
    return {};
  }

  const std::streamsize size = file.tellg();
  std::vector<char> buffer(static_cast<size_t>(size));
  file.seekg(0);
  file.read(buffer.data(), size);
  return buffer;
}

VkShaderModule create_shader_module(const char *path) {
  const std::vector<char> code = read_shader_file(path);
  if (code.empty())
    return VK_NULL_HANDLE;

  const VkShaderModuleCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = code.size(),
      .pCode = reinterpret_cast<const uint32_t *>(code.data()),
  };

  VkShaderModule module;
  if (vkCreateShaderModule(graphics::internal::context.device, &create_info,
                           nullptr, &module) != VK_SUCCESS) {
    std::cerr << "Failed to create shader module: " << path << '\n';
    return VK_NULL_HANDLE;
  }
  return module;
}

VkBuffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                       VmaAllocation *allocation, void **mapped) {
  auto &context = graphics::internal::context;

  VmaAllocationCreateInfo allocation_info = {};
  allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
  allocation_info.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      (mapped != nullptr ? VMA_ALLOCATION_CREATE_MAPPED_BIT : 0);

  VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };

  VkBuffer buffer;
  if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
                      &buffer, allocation, nullptr) != VK_SUCCESS) {
    std::cerr << "Failed to create buffer\n";
    return VK_NULL_HANDLE;
  }

  if (mapped != nullptr) {
    void *data = nullptr;
    vmaMapMemory(context.allocator, *allocation, &data);
    *mapped = data;
  }

  return buffer;
}

template <typename T>
bool upload_to_buffer(VkBuffer buffer, VmaAllocation allocation, const T *data,
                      size_t count) {
  auto &context = graphics::internal::context;

  void *mapped = nullptr;
  if (vmaMapMemory(context.allocator, allocation, &mapped) != VK_SUCCESS) {
    return false;
  }
  std::memcpy(mapped, data, count * sizeof(T));
  vmaUnmapMemory(context.allocator, allocation);
  vmaFlushAllocation(context.allocator, allocation, 0, VK_WHOLE_SIZE);
  return true;
}

bool create_pipeline(const char *vert_path, const char *frag_path,
                     VkPrimitiveTopology topology, VkCullModeFlags cull_mode,
                     VkPipeline *out_pipeline) {
  auto &context = graphics::internal::context;

  VkShaderModule vert_module = create_shader_module(vert_path);
  VkShaderModule frag_module = create_shader_module(frag_path);
  if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) {
    return false;
  }

  const VkPipelineShaderStageCreateInfo shader_stages[] = {
      {
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,
          .module = vert_module,
          .pName = "main",
      },
      {
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
          .module = frag_module,
          .pName = "main",
      },
  };

  const VkVertexInputBindingDescription vertex_binding = {
      .binding = 0,
      .stride = sizeof(Vertex),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
  };

  const VkVertexInputAttributeDescription vertex_attributes[] = {
      {
          .location = 0,
          .binding = 0,
          .format = VK_FORMAT_R32G32B32_SFLOAT,
          .offset = offsetof(Vertex, position),
      },
      {
          .location = 1,
          .binding = 0,
          .format = VK_FORMAT_R32G32B32_SFLOAT,
          .offset = offsetof(Vertex, normal),
      },
  };

  const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount = 2,
      .pVertexAttributeDescriptions = vertex_attributes,
  };

  const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = topology,
  };

  const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
  };

  const VkPipelineRasterizationStateCreateInfo rasterization = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = cull_mode,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
  };

  const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
  };

  const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE,
      .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS,
  };

  const VkPipelineColorBlendAttachmentState color_blend_attachment = {
      .blendEnable = VK_FALSE,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
  };

  const VkPipelineColorBlendStateCreateInfo color_blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &color_blend_attachment,
  };

  const VkDynamicState dynamic_states[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
  };

  const VkPipelineDynamicStateCreateInfo dynamic_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = dynamic_states,
  };

  const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = shader_stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &rasterization,
      .pMultisampleState = &multisample,
      .pDepthStencilState = &depth_stencil,
      .pColorBlendState = &color_blend,
      .pDynamicState = &dynamic_state,
      .layout = vk_pipeline_layout,
      .renderPass = context.render_pass,
      .subpass = 0,
  };

  const VkResult result = vkCreateGraphicsPipelines(
      context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, out_pipeline);

  vkDestroyShaderModule(context.device, vert_module, nullptr);
  vkDestroyShaderModule(context.device, frag_module, nullptr);

  if (result != VK_SUCCESS) {
    std::cerr << "Failed to create graphics pipeline\n";
    return false;
  }

  return true;
}

bool create_graphics_pipelines() {
  auto &context = graphics::internal::context;

  const VkPipelineLayoutCreateInfo pipeline_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 2,
      .pSetLayouts = vk_set_layouts,
  };

  if (vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr,
                             &vk_pipeline_layout) != VK_SUCCESS) {
    std::cerr << "Failed to create pipeline layout\n";
    return false;
  }

  if (!create_pipeline("shaders/lighting.vert.spv", "shaders/lighting.frag.spv",
                       VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                       VK_CULL_MODE_BACK_BIT, &vk_pipeline_lit)) {
    return false;
  }

  if (!create_pipeline("shaders/lighting.vert.spv", "shaders/unlit.frag.spv",
                       VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                       VK_CULL_MODE_BACK_BIT, &vk_pipeline_unlit)) {
    return false;
  }

  if (!create_pipeline("shaders/lighting.vert.spv", "shaders/unlit.frag.spv",
                       VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_CULL_MODE_NONE,
                       &vk_pipeline_lines)) {
    return false;
  }

  return true;
}

void write_buffer_descriptor(VkDescriptorSet set, uint32_t binding,
                             VkBuffer buffer, VkDeviceSize offset,
                             VkDeviceSize range) {
  const VkDescriptorBufferInfo buffer_info = {
      .buffer = buffer,
      .offset = offset,
      .range = range,
  };

  const VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = set,
      .dstBinding = binding,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .pBufferInfo = &buffer_info,
  };

  vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0,
                         nullptr);
}

bool create_descriptor_stuff() {
  auto &context = graphics::internal::context;

  const VkDescriptorSetLayoutBinding scene_bindings[] = {
      {
          .binding = 0,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .descriptorCount = 1,
          .stageFlags =
              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
          .binding = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
          .binding = 2,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
  };

  const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
      .pBindings = scene_bindings,
  };

  if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
                                  &vk_set_layouts[0]) != VK_SUCCESS) {
    std::cerr << "Failed to create scene descriptor set layout\n";
    return false;
  }

  const VkDescriptorSetLayoutBinding object_binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
  };

  const VkDescriptorSetLayoutCreateInfo object_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &object_binding,
  };

  if (vkCreateDescriptorSetLayout(context.device, &object_layout_info, nullptr,
                                  &vk_set_layouts[1]) != VK_SUCCESS) {
    std::cerr << "Failed to create object descriptor set layout\n";
    return false;
  }

  const uint32_t uniform_count =
      1 + uint32_t(objects.size()) + GIZMO_SLOT_COUNT;
  const VkDescriptorPoolSize pool_sizes[] = {
      {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
       .descriptorCount = uniform_count},
      {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2},
  };

  const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1 + uint32_t(objects.size()) + GIZMO_SLOT_COUNT,
      .poolSizeCount = 2,
      .pPoolSizes = pool_sizes,
  };

  if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
                             &vk_descriptor_pool) != VK_SUCCESS) {
    std::cerr << "Failed to create descriptor pool\n";
    return false;
  }

  {
    const VkDescriptorSetAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &vk_set_layouts[0],
    };

    if (vkAllocateDescriptorSets(context.device, &allocate_info,
                                 &vk_scene_descriptor_set) != VK_SUCCESS) {
      std::cerr << "Failed to allocate scene descriptor set\n";
      return false;
    }

    write_buffer_descriptor(vk_scene_descriptor_set, 0, vk_scene_uniform_buffer,
                            0, sizeof(SceneData));

    const VkDescriptorBufferInfo point_lights_info = {
        .buffer = vk_point_lights_buffer,
        .offset = 0,
        .range = sizeof(GpuPointLight) * MAX_LIGHTS,
    };
    const VkDescriptorBufferInfo spot_lights_info = {
        .buffer = vk_spot_lights_buffer,
        .offset = 0,
        .range = sizeof(GpuSpotLight) * MAX_LIGHTS,
    };

    const VkWriteDescriptorSet writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = vk_scene_descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &point_lights_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = vk_scene_descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &spot_lights_info,
        },
    };

    vkUpdateDescriptorSets(context.device, 2, writes, 0, nullptr);
  }

  for (auto &object : objects) {
    const VkDescriptorSetAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &vk_set_layouts[1],
    };

    if (vkAllocateDescriptorSets(context.device, &allocate_info,
                                 &object.descriptor_set) != VK_SUCCESS) {
      std::cerr << "Failed to allocate object descriptor set\n";
      return false;
    }

    write_buffer_descriptor(object.descriptor_set, 0, object.uniform_buffer, 0,
                            sizeof(ObjectData));
  }

  for (uint32_t i = 0; i < GIZMO_SLOT_COUNT; ++i) {
    const VkDescriptorSetAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = vk_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &vk_set_layouts[1],
    };

    if (vkAllocateDescriptorSets(context.device, &allocate_info,
                                 &vk_gizmo_descriptor_sets[i]) != VK_SUCCESS) {
      std::cerr << "Failed to allocate gizmo descriptor set #" << i << '\n';
      return false;
    }

    write_buffer_descriptor(vk_gizmo_descriptor_sets[i], 0,
                            vk_gizmo_uniform_buffer, i * GIZMO_SLOT_STRIDE,
                            sizeof(ObjectData));
  }

  return true;
}

void write_gizmo_slot(uint32_t slot, const glm::mat4 &model,
                      const glm::vec3 &color) {
  static_assert(sizeof(ObjectData) <= GIZMO_SLOT_STRIDE);

  const ObjectData data = {
      .model = model,
      .albedo = glm::vec4(color, 1.0f),
      .specular = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
  };

  char *dst =
      static_cast<char *>(vk_gizmo_uniform_mapped) + slot * GIZMO_SLOT_STRIDE;
  std::memcpy(dst, &data, sizeof(data));
}

void build_spot_cone_lines(const SpotLightState &light) {
  const glm::vec3 apex = light.position;
  const glm::vec3 dir = glm::normalize(light.direction);
  constexpr float DISTANCE = 1.2f;
  constexpr int SEGMENTS = 16;

  const float radius = std::tan(glm::radians(light.outer_angle)) * DISTANCE;
  const glm::vec3 center = apex + dir * DISTANCE;

  glm::vec3 side =
      glm::abs(dir.y) < 0.99f
          ? glm::normalize(glm::cross(dir, glm::vec3(0.0f, 1.0f, 0.0f)))
          : glm::vec3(1.0f, 0.0f, 0.0f);
  const glm::vec3 up = glm::normalize(glm::cross(side, dir));

  auto push_line = [&](glm::vec3 a, glm::vec3 b) {
    if (vk_lines_vertex_count + 2 > vk_lines_vertex_capacity)
      return;
    Vertex *dst =
        static_cast<Vertex *>(vk_lines_vertex_mapped) + vk_lines_vertex_count;
    dst[0] = {a, glm::vec3(0.0f)};
    dst[1] = {b, glm::vec3(0.0f)};
    vk_lines_vertex_count += 2;
  };

  for (int i = 0; i < SEGMENTS; ++i) {
    const float a0 = glm::two_pi<float>() * float(i) / SEGMENTS;
    const float a1 = glm::two_pi<float>() * float(i + 1) / SEGMENTS;

    const glm::vec3 p0 =
        center + radius * (std::cos(a0) * side + std::sin(a0) * up);
    const glm::vec3 p1 =
        center + radius * (std::cos(a1) * side + std::sin(a1) * up);

    push_line(apex, p0);
    push_line(p0, p1);
  }
}

} // namespace

bool initialize() {
  struct BoxDesc {
    glm::vec3 min, max;
  };
  const BoxDesc scene_boxes[] = {
      {{-3.0f, -0.1f, -3.0f}, {3.0f, 0.0f, 3.0f}},
      {{-3.1f, 0.0f, -3.0f}, {-3.0f, 2.5f, 3.0f}},
      {{3.0f, 0.0f, -3.0f}, {3.1f, 2.5f, 3.0f}},
      {{-3.0f, 0.0f, -3.1f}, {3.0f, 2.5f, -3.0f}},
      {{-0.5f, 0.0f, -0.5f}, {0.5f, 1.0f, 0.5f}},
      {{1.0f, 0.62f, 0.2f}, {1.8f, 0.72f, 1.0f}},
      {{1.08f, 0.0f, 0.28f}, {1.16f, 0.62f, 0.36f}},
      {{1.64f, 0.0f, 0.28f}, {1.72f, 0.62f, 0.36f}},
      {{1.08f, 0.0f, 0.84f}, {1.16f, 0.62f, 0.92f}},
      {{1.64f, 0.0f, 0.84f}, {1.72f, 0.62f, 0.92f}},
  };

  objects.resize(std::size(scene_boxes));
  for (size_t i = 0; i < objects.size(); ++i) {
    objects[i].mesh = make_box_mesh(scene_boxes[i].min, scene_boxes[i].max);
  }

  gizmo_cube.mesh = make_box_mesh({-0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, 0.5f});

  vk_scene_uniform_buffer =
      create_buffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    &vk_scene_uniform_allocation, &vk_scene_uniform_mapped);
  vk_point_lights_buffer = create_buffer(
      sizeof(GpuPointLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      &vk_point_lights_allocation, &vk_point_lights_mapped);
  vk_spot_lights_buffer = create_buffer(
      sizeof(GpuSpotLight) * MAX_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      &vk_spot_lights_allocation, &vk_spot_lights_mapped);
  if (vk_scene_uniform_buffer == VK_NULL_HANDLE ||
      vk_point_lights_buffer == VK_NULL_HANDLE ||
      vk_spot_lights_buffer == VK_NULL_HANDLE) {
    return false;
  }

  for (auto &object : objects) {
    object.vertex_buffer = create_buffer(
        object.mesh.vertices.size() * sizeof(Vertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &object.vertex_allocation, nullptr);
    object.index_buffer = create_buffer(
        object.mesh.indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &object.index_allocation, nullptr);
    object.uniform_buffer =
        create_buffer(sizeof(ObjectData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      &object.uniform_allocation, &object.uniform_mapped);

    if (object.vertex_buffer == VK_NULL_HANDLE ||
        object.index_buffer == VK_NULL_HANDLE ||
        object.uniform_buffer == VK_NULL_HANDLE) {
      return false;
    }

    if (!upload_to_buffer(object.vertex_buffer, object.vertex_allocation,
                          object.mesh.vertices.data(),
                          object.mesh.vertices.size())) {
      return false;
    }
    if (!upload_to_buffer(object.index_buffer, object.index_allocation,
                          object.mesh.indices.data(),
                          object.mesh.indices.size())) {
      return false;
    }
  }

  gizmo_cube.vertex_buffer =
      create_buffer(gizmo_cube.mesh.vertices.size() * sizeof(Vertex),
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    &gizmo_cube.vertex_allocation, nullptr);
  gizmo_cube.index_buffer = create_buffer(
      gizmo_cube.mesh.indices.size() * sizeof(uint32_t),
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &gizmo_cube.index_allocation, nullptr);
  if (gizmo_cube.vertex_buffer == VK_NULL_HANDLE ||
      gizmo_cube.index_buffer == VK_NULL_HANDLE) {
    return false;
  }
  upload_to_buffer(gizmo_cube.vertex_buffer, gizmo_cube.vertex_allocation,
                   gizmo_cube.mesh.vertices.data(),
                   gizmo_cube.mesh.vertices.size());
  upload_to_buffer(gizmo_cube.index_buffer, gizmo_cube.index_allocation,
                   gizmo_cube.mesh.indices.data(),
                   gizmo_cube.mesh.indices.size());

  vk_gizmo_uniform_buffer = create_buffer(
      GIZMO_SLOT_STRIDE * GIZMO_SLOT_COUNT, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      &vk_gizmo_uniform_allocation, &vk_gizmo_uniform_mapped);
  if (vk_gizmo_uniform_buffer == VK_NULL_HANDLE) {
    return false;
  }

  vk_lines_vertex_capacity = MAX_LIGHTS * 64;
  vk_lines_vertex_buffer =
      create_buffer(sizeof(Vertex) * vk_lines_vertex_capacity,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    &vk_lines_vertex_allocation, &vk_lines_vertex_mapped);
  if (vk_lines_vertex_buffer == VK_NULL_HANDLE) {
    return false;
  }

  if (!create_descriptor_stuff())
    return false;
  if (!create_graphics_pipelines())
    return false;

  return true;
}

void shutdown() {
  auto &context = graphics::internal::context;

  vkDeviceWaitIdle(context.device);

  vkDestroyPipeline(context.device, vk_pipeline_lit, nullptr);
  vkDestroyPipeline(context.device, vk_pipeline_unlit, nullptr);
  vkDestroyPipeline(context.device, vk_pipeline_lines, nullptr);
  vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
  vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
  vkDestroyDescriptorSetLayout(context.device, vk_set_layouts[0], nullptr);
  vkDestroyDescriptorSetLayout(context.device, vk_set_layouts[1], nullptr);

  vmaDestroyBuffer(context.allocator, vk_scene_uniform_buffer,
                   vk_scene_uniform_allocation);
  vmaDestroyBuffer(context.allocator, vk_point_lights_buffer,
                   vk_point_lights_allocation);
  vmaDestroyBuffer(context.allocator, vk_spot_lights_buffer,
                   vk_spot_lights_allocation);
  vmaDestroyBuffer(context.allocator, vk_gizmo_uniform_buffer,
                   vk_gizmo_uniform_allocation);
  vmaDestroyBuffer(context.allocator, vk_lines_vertex_buffer,
                   vk_lines_vertex_allocation);

  vmaDestroyBuffer(context.allocator, gizmo_cube.vertex_buffer,
                   gizmo_cube.vertex_allocation);
  vmaDestroyBuffer(context.allocator, gizmo_cube.index_buffer,
                   gizmo_cube.index_allocation);

  for (auto &object : objects) {
    vmaDestroyBuffer(context.allocator, object.vertex_buffer,
                     object.vertex_allocation);
    vmaDestroyBuffer(context.allocator, object.index_buffer,
                     object.index_allocation);
    vmaDestroyBuffer(context.allocator, object.uniform_buffer,
                     object.uniform_allocation);
  }

  objects.clear();
}

void update([[maybe_unused]] double time) {
  auto &camera = state.camera;

  ImGui::Begin("Camera");
  ImGui::DragFloat3("Position", &camera.position.x, 0.02f);
  ImGui::SliderFloat("Yaw", &camera.yaw, -180.0f, 180.0f, "%.0f deg");
  ImGui::SliderFloat("Pitch", &camera.pitch, -89.0f, 89.0f, "%.0f deg");
  ImGui::SliderFloat("FOV", &camera.fov, 20.0f, 120.0f, "%.0f deg");
  if (ImGui::Button("Reset camera")) {
    camera = CameraState{};
  }
  ImGui::End();

  ImGui::Begin("Lighting");

  ImGui::SeparatorText("Ambient");
  ImGui::ColorEdit3("Ambient color", &state.ambient_color.x);
  ImGui::SliderFloat("Ambient intensity", &state.ambient_intensity, 0.0f, 1.0f);

  ImGui::SeparatorText("Directional (sun)");
  ImGui::DragFloat3("Sun direction", &state.sun_direction.x, 0.02f);
  ImGui::ColorEdit3("Sun color", &state.sun_color.x);
  ImGui::SliderFloat("Sun intensity", &state.sun_intensity, 0.0f, 4.0f);

  ImGui::SeparatorText("Point lights");

  for (size_t i = 0; i < state.point_lights.size(); ++i) {
    auto &light = state.point_lights[i];

    if (ImGui::TreeNode(reinterpret_cast<void *>(i), "Point light #%zu", i)) {
      ImGui::DragFloat3("Position", &light.position.x, 0.02f);
      ImGui::ColorEdit3("Color", &light.color.x);
      ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 20.0f);
      if (ImGui::Button("Remove")) {
        state.point_lights.erase(state.point_lights.begin() + ptrdiff_t(i));
        ImGui::TreePop();
        break;
      }
      ImGui::TreePop();
    }
  }

  if (state.point_lights.size() < MAX_LIGHTS) {
    if (ImGui::Button("Add point light")) {
      state.point_lights.push_back(PointLightState{});
    }
  } else {
    ImGui::TextDisabled("Max %d point lights", MAX_LIGHTS);
  }

  ImGui::SeparatorText("Spot lights");

  for (size_t i = 0; i < state.spot_lights.size(); ++i) {
    auto &light = state.spot_lights[i];

    light.outer_angle = std::max(light.outer_angle, light.inner_angle);

    if (ImGui::TreeNode(reinterpret_cast<void *>(i + 64), "Spot light #%zu",
                        i)) {
      ImGui::DragFloat3("Position", &light.position.x, 0.02f);
      ImGui::DragFloat3("Direction", &light.direction.x, 0.02f);
      ImGui::ColorEdit3("Color", &light.color.x);
      ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 20.0f);
      ImGui::SliderFloat("Inner angle", &light.inner_angle, 1.0f, 85.0f,
                         "%.0f deg");
      ImGui::SliderFloat("Outer angle", &light.outer_angle, 1.0f, 85.0f,
                         "%.0f deg");
      if (ImGui::Button("Remove")) {
        state.spot_lights.erase(state.spot_lights.begin() + ptrdiff_t(i));
        ImGui::TreePop();
        break;
      }
      ImGui::TreePop();
    }
  }

  if (state.spot_lights.size() < MAX_LIGHTS) {
    if (ImGui::Button("Add spot light")) {
      state.spot_lights.push_back(SpotLightState{});
    }
  } else {
    ImGui::TextDisabled("Max %d spot lights", MAX_LIGHTS);
  }

  ImGui::Checkbox("Show light gizmos", &state.show_light_gizmos);
  ImGui::End();

  ImGui::Begin("Material");
  ImGui::ColorEdit3("Albedo", &state.material.albedo.x);
  ImGui::ColorEdit3("Specular color", &state.material.specular.x);
  ImGui::SliderFloat("Shininess", &state.material.shininess, 1.0f, 256.0f,
                     "%.0f");
  ImGui::End();
}

void render(const graphics::internal::FrameData &fd) {
  auto &context = graphics::internal::context;

  const float aspect = float(context.swapchain_extent.width) /
                       float(context.swapchain_extent.height);

  const float yaw = glm::radians(state.camera.yaw);
  const float pitch = glm::radians(state.camera.pitch);
  const glm::vec3 forward{
      std::cos(pitch) * std::cos(yaw),
      std::sin(pitch),
      std::cos(pitch) * std::sin(yaw),
  };

  SceneData scene_data = {
      .view =
          glm::lookAt(state.camera.position, state.camera.position + forward,
                      glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = glm::perspective(glm::radians(state.camera.fov), aspect, 0.05f,
                               100.0f),
      .camera_pos = glm::vec4(state.camera.position, 1.0f),
      .ambient = glm::vec4(state.ambient_color, state.ambient_intensity),
      .sun_dir = glm::vec4(glm::normalize(state.sun_direction), 0.0f),
      .sun_color = glm::vec4(state.sun_color, state.sun_intensity),
      .counts = glm::ivec4(int(state.point_lights.size()),
                           int(state.spot_lights.size()), 0, 0),
  };
  scene_data.proj[1][1] *= -1.0f;

  std::memcpy(vk_scene_uniform_mapped, &scene_data, sizeof(scene_data));
  vmaFlushAllocation(context.allocator, vk_scene_uniform_allocation, 0,
                     VK_WHOLE_SIZE);

  for (size_t i = 0; i < state.point_lights.size(); ++i) {
    const auto &light = state.point_lights[i];
    const GpuPointLight gpu_light = {
        .position = glm::vec4(light.position, 1.0f),
        .color = glm::vec4(light.color, light.intensity),
    };
    static_cast<GpuPointLight *>(vk_point_lights_mapped)[i] = gpu_light;
  }

  for (size_t i = 0; i < state.spot_lights.size(); ++i) {
    const auto &light = state.spot_lights[i];
    const GpuSpotLight gpu_light = {
        .position = glm::vec4(light.position, 1.0f),
        .direction = glm::vec4(glm::normalize(light.direction), 0.0f),
        .color = glm::vec4(light.color, light.intensity),
        .params =
            glm::vec4(std::cos(glm::radians(light.inner_angle)),
                      std::cos(glm::radians(light.outer_angle)), 0.0f, 0.0f),
    };
    static_cast<GpuSpotLight *>(vk_spot_lights_mapped)[i] = gpu_light;
  }

  vmaFlushAllocation(context.allocator, vk_point_lights_allocation, 0,
                     VK_WHOLE_SIZE);
  vmaFlushAllocation(context.allocator, vk_spot_lights_allocation, 0,
                     VK_WHOLE_SIZE);

  const ObjectData object_data = {
      .model = glm::mat4(1.0f),
      .albedo = glm::vec4(state.material.albedo, 1.0f),
      .specular = glm::vec4(state.material.specular, state.material.shininess),
  };

  for (auto &object : objects) {
    std::memcpy(object.uniform_mapped, &object_data, sizeof(object_data));
    vmaFlushAllocation(context.allocator, object.uniform_allocation, 0,
                       VK_WHOLE_SIZE);
  }

  const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
  };
  vkBeginCommandBuffer(fd.command_buffer, &begin_info);

  const VkClearValue clear_values[] = {
      {.color = {0.02f, 0.02f, 0.03f, 1.0f}},
      {.depthStencil = {1.0f, 0}},
  };

  const VkRenderPassBeginInfo render_pass_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = context.render_pass,
      .framebuffer = fd.framebuffer,
      .renderArea = {.extent = context.swapchain_extent},
      .clearValueCount = 2,
      .pClearValues = clear_values,
  };

  vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin,
                       VK_SUBPASS_CONTENTS_INLINE);

  const VkViewport viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = float(context.swapchain_extent.width),
      .height = float(context.swapchain_extent.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

  const VkRect2D scissor = {
      .extent = context.swapchain_extent,
  };
  vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

  vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    vk_pipeline_lit);

  for (const auto &object : objects) {
    const VkDescriptorSet descriptor_sets[] = {
        vk_scene_descriptor_set,
        object.descriptor_set,
    };
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            vk_pipeline_layout, 0, 2, descriptor_sets, 0,
                            nullptr);

    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &object.vertex_buffer,
                           &offset);
    vkCmdBindIndexBuffer(fd.command_buffer, object.index_buffer, 0,
                         VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(fd.command_buffer, uint32_t(object.mesh.indices.size()), 1,
                     0, 0, 0);
  }

  if (state.show_light_gizmos) {
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &gizmo_cube.vertex_buffer,
                           &offset);
    vkCmdBindIndexBuffer(fd.command_buffer, gizmo_cube.index_buffer, 0,
                         VK_INDEX_TYPE_UINT32);

    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      vk_pipeline_unlit);

    for (size_t i = 0; i < state.point_lights.size(); ++i) {
      const auto &light = state.point_lights[i];
      const glm::mat4 model = glm::translate(glm::mat4(1.0f), light.position) *
                              glm::scale(glm::mat4(1.0f), glm::vec3(0.12f));

      write_gizmo_slot(uint32_t(i), model, light.color);

      const VkDescriptorSet descriptor_sets[] = {
          vk_scene_descriptor_set,
          vk_gizmo_descriptor_sets[i],
      };
      vkCmdBindDescriptorSets(
          fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
          vk_pipeline_layout, 0, 2, descriptor_sets, 0, nullptr);

      vkCmdDrawIndexed(fd.command_buffer,
                       uint32_t(gizmo_cube.mesh.indices.size()), 1, 0, 0, 0);
    }

    for (size_t i = 0; i < state.spot_lights.size(); ++i) {
      const auto &light = state.spot_lights[i];
      const glm::mat4 model = glm::translate(glm::mat4(1.0f), light.position) *
                              glm::scale(glm::mat4(1.0f), glm::vec3(0.12f));

      write_gizmo_slot(MAX_LIGHTS + uint32_t(i), model, light.color);

      const VkDescriptorSet descriptor_sets[] = {
          vk_scene_descriptor_set,
          vk_gizmo_descriptor_sets[MAX_LIGHTS + i],
      };
      vkCmdBindDescriptorSets(
          fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
          vk_pipeline_layout, 0, 2, descriptor_sets, 0, nullptr);

      vkCmdDrawIndexed(fd.command_buffer,
                       uint32_t(gizmo_cube.mesh.indices.size()), 1, 0, 0, 0);
    }

    vk_lines_vertex_count = 0;
    for (const auto &light : state.spot_lights) {
      build_spot_cone_lines(light);
    }

    if (vk_lines_vertex_count > 0) {
      vmaFlushAllocation(context.allocator, vk_lines_vertex_allocation, 0,
                         VK_WHOLE_SIZE);

      write_gizmo_slot(GIZMO_SLOT_COUNT - 1, glm::mat4(1.0f), glm::vec3(1.0f));

      const VkDescriptorSet descriptor_sets[] = {
          vk_scene_descriptor_set,
          vk_gizmo_descriptor_sets[GIZMO_SLOT_COUNT - 1],
      };
      vkCmdBindDescriptorSets(
          fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
          vk_pipeline_layout, 0, 2, descriptor_sets, 0, nullptr);

      vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        vk_pipeline_lines);
      vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vk_lines_vertex_buffer,
                             &offset);

      vkCmdDraw(fd.command_buffer, vk_lines_vertex_count, 1, 0, 0);
    }

    vmaFlushAllocation(context.allocator, vk_gizmo_uniform_allocation, 0,
                       VK_WHOLE_SIZE);
  }

  vkCmdEndRenderPass(fd.command_buffer);

  vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
