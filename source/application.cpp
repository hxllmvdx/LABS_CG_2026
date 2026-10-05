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
#include <vector>

namespace application {

namespace {

struct Vertex {
  glm::vec3 position;
  glm::vec3 color;
};

struct GlobalTransforms {
  glm::mat4 view;
  glm::mat4 proj;
};

struct ObjectTransforms {
  glm::mat4 model;
  alignas(16) glm::vec4 tint;
};

struct Mesh {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
};

struct GpuObject {
  Mesh mesh;

  VkBuffer vertex_buffer;
  VmaAllocation vertex_allocation;
  VkBuffer index_buffer;
  VmaAllocation index_allocation;

  VkBuffer uniform_buffer;
  VmaAllocation uniform_allocation;
  void *uniform_mapped = nullptr;

  VkDescriptorSet descriptor_set;
};

Mesh make_cube_mesh() {
  const glm::vec3 p[8] = {
      {-0.5f, -0.5f, -0.5f}, {+0.5f, -0.5f, -0.5f}, {+0.5f, +0.5f, -0.5f},
      {-0.5f, +0.5f, -0.5f}, {-0.5f, -0.5f, +0.5f}, {+0.5f, -0.5f, +0.5f},
      {+0.5f, +0.5f, +0.5f}, {-0.5f, +0.5f, +0.5f},
  };

  Mesh mesh;

  auto face = [&](int a, int b, int c, int d) {
    const uint32_t base = uint32_t(mesh.vertices.size());

    for (int i : {a, b, c, d}) {
      const glm::vec3 color = p[i] * 0.5f + 0.5f;
      mesh.vertices.push_back({p[i], color});
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

  face(4, 5, 6, 7);
  face(1, 0, 3, 2);
  face(5, 1, 2, 6);
  face(0, 4, 7, 3);
  face(7, 6, 2, 3);
  face(0, 1, 5, 4);

  return mesh;
}

struct AppState {
  bool orthographic = false;

  glm::vec3 position{0.0f, 0.0f, 0.0f};
  glm::vec3 rotation{0.0f, 0.0f, 0.0f};
  glm::vec3 scale{1.0f, 1.0f, 1.0f};

  bool playing = true;
  float speed = 1.0f;
  float trajectory_radius = 1.5f;
  float animation_time = 0.0f;

  glm::vec3 tint{1.0f, 1.0f, 1.0f};
};

AppState state;
double last_update_time = 0.0;

VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline;
VkDescriptorSetLayout vk_set_layouts[2];
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_global_descriptor_set;

VkBuffer vk_global_uniform_buffer;
VmaAllocation vk_global_uniform_allocation;
void *vk_global_uniform_mapped = nullptr;

std::vector<GpuObject> objects;

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

bool create_graphics_pipeline() {
  auto &context = graphics::internal::context;

  VkShaderModule vert_module = create_shader_module("shaders/cube.vert.spv");
  VkShaderModule frag_module = create_shader_module("shaders/cube.frag.spv");
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
          .offset = offsetof(Vertex, color),
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
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
  };

  const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
  };

  const VkPipelineRasterizationStateCreateInfo rasterization = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_BACK_BIT,
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
      context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &vk_pipeline);

  vkDestroyShaderModule(context.device, vert_module, nullptr);
  vkDestroyShaderModule(context.device, frag_module, nullptr);

  if (result != VK_SUCCESS) {
    std::cerr << "Failed to create graphics pipeline\n";
    return false;
  }

  return true;
}

bool create_descriptor_stuff() {
  auto &context = graphics::internal::context;

  const VkDescriptorSetLayoutBinding global_binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
  };

  const VkDescriptorSetLayoutBinding object_binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
  };

  const VkDescriptorSetLayoutCreateInfo set_layouts_info[2] = {
      {
          .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
          .bindingCount = 1,
          .pBindings = &global_binding,
      },
      {
          .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
          .bindingCount = 1,
          .pBindings = &object_binding,
      },
  };

  for (int i = 0; i < 2; ++i) {
    if (vkCreateDescriptorSetLayout(context.device, &set_layouts_info[i],
                                    nullptr,
                                    &vk_set_layouts[i]) != VK_SUCCESS) {
      std::cerr << "Failed to create descriptor set layout #" << i << '\n';
      return false;
    }
  }

  const VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = uint32_t(1 + objects.size()),
  };

  const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = uint32_t(1 + objects.size()),
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
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
                                 &vk_global_descriptor_set) != VK_SUCCESS) {
      std::cerr << "Failed to allocate global descriptor set\n";
      return false;
    }

    const VkDescriptorBufferInfo buffer_info = {
        .buffer = vk_global_uniform_buffer,
        .offset = 0,
        .range = sizeof(GlobalTransforms),
    };

    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = vk_global_descriptor_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &buffer_info,
    };

    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
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

    const VkDescriptorBufferInfo buffer_info = {
        .buffer = object.uniform_buffer,
        .offset = 0,
        .range = sizeof(ObjectTransforms),
    };

    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = object.descriptor_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &buffer_info,
    };

    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
  }

  return true;
}

} // namespace

bool initialize() {
  objects.resize(2);
  objects[0].mesh = make_cube_mesh();
  objects[1].mesh = make_cube_mesh();

  vk_global_uniform_buffer = create_buffer(
      sizeof(GlobalTransforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      &vk_global_uniform_allocation, &vk_global_uniform_mapped);
  if (vk_global_uniform_buffer == VK_NULL_HANDLE)
    return false;

  for (auto &object : objects) {
    object.vertex_buffer = create_buffer(
        object.mesh.vertices.size() * sizeof(Vertex),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &object.vertex_allocation, nullptr);
    object.index_buffer = create_buffer(
        object.mesh.indices.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &object.index_allocation, nullptr);
    object.uniform_buffer = create_buffer(
        sizeof(ObjectTransforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
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

  if (!create_descriptor_stuff())
    return false;
  if (!create_graphics_pipeline())
    return false;

  return true;
}

void shutdown() {
  auto &context = graphics::internal::context;

  vkDeviceWaitIdle(context.device);

  vkDestroyPipeline(context.device, vk_pipeline, nullptr);
  vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
  vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
  vkDestroyDescriptorSetLayout(context.device, vk_set_layouts[0], nullptr);
  vkDestroyDescriptorSetLayout(context.device, vk_set_layouts[1], nullptr);

  vmaDestroyBuffer(context.allocator, vk_global_uniform_buffer,
                   vk_global_uniform_allocation);

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
  const double dt = time - last_update_time;
  last_update_time = time;

  if (state.playing) {
    state.animation_time += float(dt) * state.speed;
  }

  ImGui::Begin("Lab 1: Cube");

  ImGui::Text("Projection");
  int projection_kind = state.orthographic ? 1 : 0;
  if (ImGui::RadioButton("Perspective", &projection_kind, 0))
    state.orthographic = false;
  ImGui::SameLine();
  if (ImGui::RadioButton("Orthographic", &projection_kind, 1))
    state.orthographic = true;

  ImGui::SeparatorText("Transform");
  ImGui::DragFloat3("Position", &state.position.x, 0.02f, -5.0f, 5.0f);
  ImGui::DragFloat3("Rotation", &state.rotation.x, 1.0f, 0.0f, 360.0f,
                    "%.0f deg");
  ImGui::DragFloat3("Scale", &state.scale.x, 0.02f, 0.1f, 5.0f);

  ImGui::SeparatorText("Animation");

  if (ImGui::Button(state.playing ? "Pause" : "Play")) {
    state.playing = !state.playing;
  }

  ImGui::SameLine();

  if (ImGui::Button("Reset")) {
    state.animation_time = 0.0f;
  }

  ImGui::SliderFloat("Speed", &state.speed, 0.0f, 3.0f);
  ImGui::SliderFloat("Trajectory radius", &state.trajectory_radius, 0.5f, 3.0f);

  ImGui::SeparatorText("Color");
  ImGui::ColorEdit3("Tint", &state.tint.x);

  ImGui::End();
}

glm::vec3 trajectory_point(float t, float radius, float phase) {
  return {
      radius * std::cos(t + phase),
      0.6f * radius * std::sin(2.0f * t + phase),
      radius * std::sin(t + phase),
  };
}

void render(const graphics::internal::FrameData &fd) {
  auto &context = graphics::internal::context;

  const float aspect = float(context.swapchain_extent.width) /
                       float(context.swapchain_extent.height);

  GlobalTransforms global_transforms = {
      .view = glm::lookAt(glm::vec3(2.5f, 2.0f, 3.5f), glm::vec3(0.0f),
                          glm::vec3(0.0f, 1.0f, 0.0f)),
      .proj = state.orthographic
                  ? glm::ortho(-1.6f * aspect, 1.6f * aspect, -1.6f, 1.6f, 0.1f,
                               100.0f)
                  : glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f),
  };

  global_transforms.proj[1][1] *= -1.0f;

  std::memcpy(vk_global_uniform_mapped, &global_transforms,
              sizeof(global_transforms));
  vmaFlushAllocation(context.allocator, vk_global_uniform_allocation, 0,
                     VK_WHOLE_SIZE);

  for (size_t i = 0; i < objects.size(); ++i) {
    const float phase = float(i) * glm::pi<float>();

    glm::mat4 model = glm::translate(
        glm::mat4(1.0f),
        state.position + trajectory_point(state.animation_time,
                                          state.trajectory_radius, phase));

    model = glm::rotate(model, glm::radians(state.rotation.x),
                        glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::rotate(model, glm::radians(state.rotation.y),
                        glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::rotate(model, glm::radians(state.rotation.z),
                        glm::vec3(0.0f, 0.0f, 1.0f));

    model = glm::scale(model, state.scale * (i == 0 ? 1.0f : 0.4f));

    const ObjectTransforms object_transforms = {
        .model = model,
        .tint = glm::vec4(state.tint, 1.0f),
    };

    std::memcpy(objects[i].uniform_mapped, &object_transforms,
                sizeof(object_transforms));
    vmaFlushAllocation(context.allocator, objects[i].uniform_allocation, 0,
                       VK_WHOLE_SIZE);
  }

  const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
  };
  vkBeginCommandBuffer(fd.command_buffer, &begin_info);

  const VkClearValue clear_values[] = {
      {.color = {0.1f, 0.1f, 0.12f, 1.0f}},
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
                    vk_pipeline);

  for (const auto &object : objects) {
    const VkDescriptorSet descriptor_sets[] = {
        vk_global_descriptor_set,
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

  vkCmdEndRenderPass(fd.command_buffer);

  vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
