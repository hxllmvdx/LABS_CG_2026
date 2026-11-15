#include "application.hpp"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../shaders/stb_image.h"

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
  glm::vec2 uv;
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
  alignas(16) glm::vec4 effect;
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
  int texture_index = 0;
  bool liquid_glass = false;

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

    const glm::vec2 uvs[] = {{0.0f, 0.0f}, {1.0f, 0.0f},
                             {1.0f, 1.0f}, {0.0f, 1.0f}};
    int corner = 0;
    for (int i : {a, b, c, d}) {
      mesh.vertices.push_back({p[i], normal, uvs[corner++]});
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
  float texture_scale = 1.0f;
  float wave_amplitude = 0.025f;
  float wave_frequency = 18.0f;
  float wave_speed = 2.0f;
};

struct PostProcessState {
  bool toon = true;
  bool bloom = true;
  float bloom_threshold = 0.8f;
  float bloom_intensity = 1.2f;
  float bloom_radius = 6.0f;
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
  PostProcessState post_process;

  bool show_light_gizmos = true;
};

AppState state;
double animation_time = 0.0;

VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline_lit;
VkPipeline vk_pipeline_unlit;
VkPipeline vk_pipeline_lines;
VkPipeline vk_pipeline_post;

VkDescriptorSetLayout vk_set_layouts[3];
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_scene_descriptor_set;
VkDescriptorSet vk_post_descriptor_set;

VkImage vk_scene_color_image = VK_NULL_HANDLE;
VmaAllocation vk_scene_color_allocation = nullptr;
VkImageView vk_scene_color_view = VK_NULL_HANDLE;
VkImage vk_scene_depth_image = VK_NULL_HANDLE;
VmaAllocation vk_scene_depth_allocation = nullptr;
VkImageView vk_scene_depth_view = VK_NULL_HANDLE;
VkFramebuffer vk_scene_framebuffer = VK_NULL_HANDLE;
VkRenderPass vk_scene_render_pass = VK_NULL_HANDLE;
VkFormat vk_scene_depth_format = VK_FORMAT_UNDEFINED;
VkSampler vk_scene_sampler = VK_NULL_HANDLE;
VkExtent2D vk_scene_extent{};

struct Texture {
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation allocation = nullptr;
  VkImageView view = VK_NULL_HANDLE;
};
std::vector<Texture> textures;

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

bool load_texture(const char *path, Texture &texture) {
  auto &context = graphics::internal::context;
  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_uc *pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
  if (pixels == nullptr) {
    std::cerr << "Failed to load texture: " << path << '\n';
    return false;
  }

  const VkDeviceSize size = VkDeviceSize(width) * VkDeviceSize(height) * 4;
  VmaAllocation staging_allocation = nullptr;
  VkBuffer staging = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                   &staging_allocation, nullptr);
  if (staging == VK_NULL_HANDLE) {
    stbi_image_free(pixels);
    return false;
  }
  void *mapped = nullptr;
  vmaMapMemory(context.allocator, staging_allocation, &mapped);
  std::memcpy(mapped, pixels, size);
  vmaFlushAllocation(context.allocator, staging_allocation, 0, VK_WHOLE_SIZE);
  vmaUnmapMemory(context.allocator, staging_allocation);
  stbi_image_free(pixels);

  const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {uint32_t(width), uint32_t(height), 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  VmaAllocationCreateInfo allocation_info = {};
  allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
  if (vmaCreateImage(context.allocator, &image_info, &allocation_info,
                     &texture.image, &texture.allocation, nullptr) != VK_SUCCESS)
    return false;

  const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = context.graphics_queue_index,
  };
  VkCommandPool pool = VK_NULL_HANDLE;
  vkCreateCommandPool(context.device, &pool_info, nullptr, &pool);
  const VkCommandBufferAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  VkCommandBuffer command_buffer = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(context.device, &allocate_info, &command_buffer);
  const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
  };
  vkBeginCommandBuffer(command_buffer, &begin_info);

  VkImageMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = texture.image,
      .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                           .levelCount = 1,
                           .layerCount = 1},
  };
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                       1, &barrier);
  const VkBufferImageCopy copy = {
      .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                           .layerCount = 1},
      .imageExtent = {uint32_t(width), uint32_t(height), 1},
  };
  vkCmdCopyBufferToImage(command_buffer, staging, texture.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);
  vkEndCommandBuffer(command_buffer);
  const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
  };
  vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(context.graphics_queue);
  vkDestroyCommandPool(context.device, pool, nullptr);
  vmaDestroyBuffer(context.allocator, staging, staging_allocation);

  const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = texture.image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                           .levelCount = 1,
                           .layerCount = 1},
  };
  return vkCreateImageView(context.device, &view_info, nullptr, &texture.view) ==
         VK_SUCCESS;
}

void destroy_scene_target() {
  auto &context = graphics::internal::context;
  if (vk_scene_framebuffer != VK_NULL_HANDLE)
    vkDestroyFramebuffer(context.device, vk_scene_framebuffer, nullptr);
  if (vk_scene_depth_view != VK_NULL_HANDLE)
    vkDestroyImageView(context.device, vk_scene_depth_view, nullptr);
  if (vk_scene_color_view != VK_NULL_HANDLE)
    vkDestroyImageView(context.device, vk_scene_color_view, nullptr);
  if (vk_scene_depth_image != VK_NULL_HANDLE)
    vmaDestroyImage(context.allocator, vk_scene_depth_image,
                    vk_scene_depth_allocation);
  if (vk_scene_color_image != VK_NULL_HANDLE)
    vmaDestroyImage(context.allocator, vk_scene_color_image,
                    vk_scene_color_allocation);
  vk_scene_framebuffer = VK_NULL_HANDLE;
  vk_scene_depth_view = VK_NULL_HANDLE;
  vk_scene_color_view = VK_NULL_HANDLE;
  vk_scene_depth_image = VK_NULL_HANDLE;
  vk_scene_color_image = VK_NULL_HANDLE;
}

bool create_scene_render_pass() {
  auto &context = graphics::internal::context;
  for (VkFormat candidate : {VK_FORMAT_D24_UNORM_S8_UINT,
                             VK_FORMAT_D32_SFLOAT_S8_UINT}) {
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(context.physical_device, candidate,
                                        &properties);
    if (properties.optimalTilingFeatures &
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
      vk_scene_depth_format = candidate;
      break;
    }
  }
  if (vk_scene_depth_format == VK_FORMAT_UNDEFINED)
    return false;

  const VkAttachmentDescription attachments[] = {
      {
          .format = context.swapchain_format,
          .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
          .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      },
      {
          .format = vk_scene_depth_format,
          .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
          .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      },
  };
  const VkAttachmentReference color_reference = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
  };
  const VkAttachmentReference depth_reference = {
      .attachment = 1,
      .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
  };
  const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_reference,
      .pDepthStencilAttachment = &depth_reference,
  };
  const VkSubpassDependency dependencies[] = {
      {
          .srcSubpass = VK_SUBPASS_EXTERNAL,
          .dstSubpass = 0,
          .srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
          .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                          VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
          .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
          .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                           VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      },
      {
          .srcSubpass = 0,
          .dstSubpass = VK_SUBPASS_EXTERNAL,
          .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
          .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
          .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      },
  };
  const VkRenderPassCreateInfo create_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 2,
      .pAttachments = attachments,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 2,
      .pDependencies = dependencies,
  };
  return vkCreateRenderPass(context.device, &create_info, nullptr,
                            &vk_scene_render_pass) == VK_SUCCESS;
}

bool create_scene_target(VkExtent2D extent) {
  auto &context = graphics::internal::context;
  destroy_scene_target();

  const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = context.swapchain_format,
      .extent = {extent.width, extent.height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  VmaAllocationCreateInfo allocation_info = {};
  allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
  if (vmaCreateImage(context.allocator, &image_info, &allocation_info,
                     &vk_scene_color_image, &vk_scene_color_allocation,
                     nullptr) != VK_SUCCESS)
    return false;

  VkImageCreateInfo depth_info = image_info;
  depth_info.format = vk_scene_depth_format;
  depth_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  if (vmaCreateImage(context.allocator, &depth_info, &allocation_info,
                     &vk_scene_depth_image, &vk_scene_depth_allocation,
                     nullptr) != VK_SUCCESS)
    return false;

  VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = vk_scene_color_image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = context.swapchain_format,
      .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                           .levelCount = 1,
                           .layerCount = 1},
  };
  if (vkCreateImageView(context.device, &view_info, nullptr,
                        &vk_scene_color_view) != VK_SUCCESS)
    return false;
  view_info.image = vk_scene_depth_image;
  view_info.format = depth_info.format;
  view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  if (vkCreateImageView(context.device, &view_info, nullptr,
                        &vk_scene_depth_view) != VK_SUCCESS)
    return false;

  const VkImageView attachments[] = {vk_scene_color_view, vk_scene_depth_view};
  const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = vk_scene_render_pass,
      .attachmentCount = 2,
      .pAttachments = attachments,
      .width = extent.width,
      .height = extent.height,
      .layers = 1,
  };
  if (vkCreateFramebuffer(context.device, &framebuffer_info, nullptr,
                          &vk_scene_framebuffer) != VK_SUCCESS)
    return false;

  const VkDescriptorImageInfo descriptor_image = {
      .sampler = vk_scene_sampler,
      .imageView = vk_scene_color_view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
  };
  const VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = vk_post_descriptor_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &descriptor_image,
  };
  vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
  vk_scene_extent = extent;
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
      {
          .location = 2,
          .binding = 0,
          .format = VK_FORMAT_R32G32_SFLOAT,
          .offset = offsetof(Vertex, uv),
      },
  };

  const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount = 3,
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
      .renderPass = out_pipeline == &vk_pipeline_post ? context.render_pass
                                                     : vk_scene_render_pass,
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
      .setLayoutCount = 3,
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

  if (!create_pipeline("shaders/lighting.vert.spv", "shaders/post.frag.spv",
                       VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_CULL_MODE_NONE,
                       &vk_pipeline_post)) {
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

  const VkDescriptorSetLayoutBinding object_bindings[] = {
      {
          .binding = 0,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .descriptorCount = 1,
          .stageFlags =
              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
          .binding = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
  };

  const VkDescriptorSetLayoutCreateInfo object_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = object_bindings,
  };

  if (vkCreateDescriptorSetLayout(context.device, &object_layout_info, nullptr,
                                  &vk_set_layouts[1]) != VK_SUCCESS) {
    std::cerr << "Failed to create object descriptor set layout\n";
    return false;
  }

  const VkDescriptorSetLayoutBinding post_binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
  };
  const VkDescriptorSetLayoutCreateInfo post_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &post_binding,
  };
  if (vkCreateDescriptorSetLayout(context.device, &post_layout_info, nullptr,
                                  &vk_set_layouts[2]) != VK_SUCCESS)
    return false;

  const uint32_t uniform_count =
      1 + uint32_t(objects.size()) + GIZMO_SLOT_COUNT;
  const VkDescriptorPoolSize pool_sizes[] = {
      {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
       .descriptorCount = uniform_count},
      {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2},
      {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount =
           1 + uint32_t(objects.size()) + GIZMO_SLOT_COUNT},
  };

  const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2 + uint32_t(objects.size()) + GIZMO_SLOT_COUNT,
      .poolSizeCount = 3,
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
        .pSetLayouts = &vk_set_layouts[2],
    };
    if (vkAllocateDescriptorSets(context.device, &allocate_info,
                                 &vk_post_descriptor_set) != VK_SUCCESS)
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
    objects[i].texture_index = int(i % 3);
  }
  objects[4].liquid_glass = true;

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
  if (!create_scene_render_pass())
    return false;

  const VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = 0.0f,
  };
  if (vkCreateSampler(graphics::internal::context.device, &sampler_info, nullptr,
                      &vk_scene_sampler) != VK_SUCCESS)
    return false;
  for (const char *path : {"assets/D.jpeg", "assets/les.jpeg",
                           "assets/meme.jpeg"}) {
    textures.emplace_back();
    if (!load_texture(path, textures.back()))
      return false;
  }
  if (!create_scene_target(graphics::internal::context.swapchain_extent))
    return false;
  if (!create_graphics_pipelines())
    return false;

  return true;
}

void shutdown() {
  auto &context = graphics::internal::context;

  vkDeviceWaitIdle(context.device);

  destroy_scene_target();
  for (auto &texture : textures) {
    vkDestroyImageView(context.device, texture.view, nullptr);
    vmaDestroyImage(context.allocator, texture.image, texture.allocation);
  }
  textures.clear();
  vkDestroySampler(context.device, vk_scene_sampler, nullptr);
  vkDestroyPipeline(context.device, vk_pipeline_lit, nullptr);
  vkDestroyPipeline(context.device, vk_pipeline_unlit, nullptr);
  vkDestroyPipeline(context.device, vk_pipeline_lines, nullptr);
  vkDestroyPipeline(context.device, vk_pipeline_post, nullptr);
  vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
  vkDestroyRenderPass(context.device, vk_scene_render_pass, nullptr);
  vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
  vkDestroyDescriptorSetLayout(context.device, vk_set_layouts[0], nullptr);
  vkDestroyDescriptorSetLayout(context.device, vk_set_layouts[1], nullptr);
  vkDestroyDescriptorSetLayout(context.device, vk_set_layouts[2], nullptr);

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

void update(double time) {
  animation_time = time;
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
  ImGui::SliderFloat("Texture scale", &state.material.texture_scale, 0.25f, 4.0f);
  ImGui::SeparatorText("Liquid glass texture waves");
  ImGui::SliderFloat("Wave amplitude", &state.material.wave_amplitude, 0.0f, 0.1f);
  ImGui::SliderFloat("Wave frequency", &state.material.wave_frequency, 1.0f, 50.0f);
  ImGui::SliderFloat("Wave speed", &state.material.wave_speed, 0.0f, 6.0f);
  ImGui::End();

  const char *texture_names[] = {"D.jpeg", "les.jpeg", "meme.jpeg"};
  ImGui::Begin("Model textures");
  for (size_t i = 0; i < objects.size(); ++i) {
    ImGui::PushID(int(i));
    const char *label = i == 4 ? "Central cube" : (i < 4 ? "Room surface" : "Table part");
    ImGui::SeparatorText(label);
    ImGui::Combo("Texture", &objects[i].texture_index, texture_names, 3);
    ImGui::Checkbox("Animated liquid glass", &objects[i].liquid_glass);
    ImGui::PopID();
  }
  ImGui::End();

  auto &post = state.post_process;
  ImGui::Begin("Post-processing");
  ImGui::Checkbox("Toon shading + outline", &post.toon);
  ImGui::SeparatorText("Bloom");
  ImGui::Checkbox("Enable bloom", &post.bloom);
  ImGui::SliderFloat("Brightness threshold", &post.bloom_threshold, 0.1f, 3.0f);
  ImGui::SliderFloat("Bloom intensity", &post.bloom_intensity, 0.0f, 4.0f);
  ImGui::SliderFloat("Bloom radius", &post.bloom_radius, 1.0f, 20.0f);
  ImGui::End();
}

void render(const graphics::internal::FrameData &fd) {
  auto &context = graphics::internal::context;

  if (vk_scene_extent.width != context.swapchain_extent.width ||
      vk_scene_extent.height != context.swapchain_extent.height) {
    vkDeviceWaitIdle(context.device);
    if (!create_scene_target(context.swapchain_extent)) {
      std::cerr << "Failed to resize post-processing target\n";
      return;
    }
  }

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
      .counts = glm::ivec4(
          int(state.point_lights.size()), int(state.spot_lights.size()),
          (state.post_process.toon ? 1 : 0) |
              (state.post_process.bloom ? 2 : 0),
          int(animation_time * 1000.0)),
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

  for (auto &object : objects) {
    const ObjectData object_data = {
        .model = glm::mat4(1.0f),
        .albedo = glm::vec4(state.material.albedo, state.material.texture_scale),
        .specular =
            glm::vec4(state.material.specular, state.material.shininess),
        .effect = glm::vec4(object.liquid_glass ? 1.0f : 0.0f,
                            state.material.wave_amplitude,
                            state.material.wave_frequency,
                            state.material.wave_speed),
    };
    std::memcpy(object.uniform_mapped, &object_data, sizeof(object_data));
    vmaFlushAllocation(context.allocator, object.uniform_allocation, 0,
                       VK_WHOLE_SIZE);

    const VkDescriptorImageInfo texture_info = {
        .sampler = vk_scene_sampler,
        .imageView = textures[size_t(object.texture_index)].view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkWriteDescriptorSet texture_write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = object.descriptor_set,
        .dstBinding = 1,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &texture_info,
    };
    vkUpdateDescriptorSets(context.device, 1, &texture_write, 0, nullptr);
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
      .renderPass = vk_scene_render_pass,
      .framebuffer = vk_scene_framebuffer,
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

  ObjectData post_data = {
      .model = glm::mat4(1.0f),
      .albedo = glm::vec4(1.0f),
      .specular = glm::vec4(0.0f, 0.0f, 0.0f, -1.0f),
  };
  post_data.model[1] = glm::vec4(state.post_process.bloom_threshold,
                                 state.post_process.bloom_intensity,
                                 state.post_process.bloom_radius, 0.0f);
  char *post_slot = static_cast<char *>(vk_gizmo_uniform_mapped) +
                    (GIZMO_SLOT_COUNT - 1) * GIZMO_SLOT_STRIDE;
  std::memcpy(post_slot, &post_data, sizeof(post_data));
  vmaFlushAllocation(context.allocator, vk_gizmo_uniform_allocation, 0,
                     VK_WHOLE_SIZE);

  const VkRenderPassBeginInfo final_pass_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = context.render_pass,
      .framebuffer = fd.framebuffer,
      .renderArea = {.extent = context.swapchain_extent},
      .clearValueCount = 2,
      .pClearValues = clear_values,
  };
  vkCmdBeginRenderPass(fd.command_buffer, &final_pass_begin,
                       VK_SUBPASS_CONTENTS_INLINE);
  vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
  vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
  vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    vk_pipeline_post);
  const VkDescriptorSet post_sets[] = {
      vk_scene_descriptor_set,
      vk_gizmo_descriptor_sets[GIZMO_SLOT_COUNT - 1],
      vk_post_descriptor_set,
  };
  vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          vk_pipeline_layout, 0, 3, post_sets, 0, nullptr);
  vkCmdDraw(fd.command_buffer, 3, 1, 0, 0);
  vkCmdEndRenderPass(fd.command_buffer);

  vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
