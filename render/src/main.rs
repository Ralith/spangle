#[cfg(feature = "renderdoc")]
use std::ffi::c_void;
use std::{mem, path::PathBuf, ptr};

use ash::vk;
#[cfg(feature = "renderdoc")]
use ash::vk::Handle;
use clap::Parser;
#[cfg(feature = "renderdoc")]
use renderdoc::RenderDoc;
use vk_shader_macros::include_glsl;

const VERT: &[u32] = include_glsl!("src/star.vert");
const FRAG: &[u32] = include_glsl!("src/star.frag");

macro_rules! cstr {
    ($x:literal) => {{
        std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($x, "\0").as_bytes())
    }};
}

#[derive(Parser)]
/// Visualize a 90 degree frustum of a starfield
struct Options {
    /// Pixels per dimension for the output image
    #[clap(short = 'r', default_value = "2048")]
    resolution: u32,
    /// Total number of stars in the sphere (only about 1/6 of these will be visible)
    #[clap(short = 'c', default_value = "20000")]
    star_count: usize,
    /// Scale factor for the rate of large-scale variation in irradiance
    #[clap(short = 'f', default_value = "4.0")]
    irradiance_frequency: f32,
    /// PNG file to write
    out: PathBuf,
}

fn main() {
    let options = Options::parse();
    let mut ctx = Context::new(&options);

    let stars = spangle::Starfield::new()
        .irradiance_frequency(options.irradiance_frequency)
        .count(options.star_count)
        .generate();

    let projection = projection(1.0);
    println!("drawing...");
    let data = ctx.draw(&projection, stars);

    println!("saving...");
    lodepng::encode24_file(
        &options.out,
        &data,
        options.resolution as usize,
        options.resolution as usize,
    )
    .unwrap();
}

struct Context {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    #[cfg(feature = "renderdoc")]
    renderdoc: Option<RenderDoc<renderdoc::V100>>,
    queue: vk::Queue,
    render_pass: vk::RenderPass,

    ds_layout: vk::DescriptorSetLayout,
    ds_pool: vk::DescriptorPool,
    ds: vk::DescriptorSet,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    // Inputs
    resolution: u32,
    star_count: usize,
    uniforms: vk::Buffer,
    uniforms_mem: vk::DeviceMemory,
    uniforms_ptr: *mut u8,
    stars: vk::Buffer,
    stars_mem: vk::DeviceMemory,
    stars_ptr: *mut u8,

    // Outputs
    image: vk::Image,
    image_mem: vk::DeviceMemory,
    image_view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    transfer: vk::Buffer,
    transfer_mem: vk::DeviceMemory,
    transfer_ptr: *mut u8,

    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
}

impl Context {
    fn new(options: &Options) -> Self {
        unsafe {
            //
            // Global setup
            //
            let entry = ash::Entry::load().unwrap();
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::builder().application_info(
                        &vk::ApplicationInfo::builder()
                            .application_name(cstr!("spangle-render"))
                            .engine_name(cstr!("spangle-render"))
                            .api_version(vk::make_api_version(0, 1, 0, 0)),
                    ),
                    None,
                )
                .unwrap();
            let physical_device = instance.enumerate_physical_devices().unwrap()[0];
            let memory_props = instance.get_physical_device_memory_properties(physical_device);
            let queue_family_index = instance
                .get_physical_device_queue_family_properties(physical_device)
                .into_iter()
                .enumerate()
                .find_map(|(index, props)| {
                    if props.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        Some(index)
                    } else {
                        None
                    }
                })
                .expect("no graphics queue") as u32;
            let device = instance
                .create_device(
                    physical_device,
                    &vk::DeviceCreateInfo::builder().queue_create_infos(&[
                        vk::DeviceQueueCreateInfo::builder()
                            .queue_family_index(queue_family_index)
                            .queue_priorities(&[1.0])
                            .build(),
                    ]),
                    None,
                )
                .unwrap();
            let queue = device.get_device_queue(queue_family_index, 0);

            let render_pass = device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::builder()
                        .attachments(&[vk::AttachmentDescription {
                            format: COLOR_FORMAT,
                            samples: vk::SampleCountFlags::TYPE_1,
                            load_op: vk::AttachmentLoadOp::CLEAR,
                            store_op: vk::AttachmentStoreOp::STORE,
                            initial_layout: vk::ImageLayout::UNDEFINED,
                            final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            ..Default::default()
                        }])
                        .subpasses(&[vk::SubpassDescription::builder()
                            .color_attachments(&[vk::AttachmentReference {
                                attachment: 0,
                                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            }])
                            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                            .build()])
                        .dependencies(&[
                            vk::SubpassDependency {
                                src_subpass: vk::SUBPASS_EXTERNAL,
                                dst_subpass: 0,
                                src_stage_mask: vk::PipelineStageFlags::TOP_OF_PIPE,
                                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                                    | vk::PipelineStageFlags::FRAGMENT_SHADER,
                                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                                ..Default::default()
                            },
                            vk::SubpassDependency {
                                src_subpass: 0,
                                dst_subpass: vk::SUBPASS_EXTERNAL,
                                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                                dst_stage_mask: vk::PipelineStageFlags::TRANSFER,
                                src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                                dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                                dependency_flags: vk::DependencyFlags::empty(),
                            },
                        ]),
                    None,
                )
                .unwrap();

            //
            // Pipeline setup
            //

            let vert = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(VERT), None)
                .unwrap();
            let frag = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(FRAG), None)
                .unwrap();

            let ds_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::VERTEX,
                            p_immutable_samplers: ptr::null(),
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::VERTEX,
                            p_immutable_samplers: ptr::null(),
                        },
                    ]),
                    None,
                )
                .unwrap();
            let ds_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::builder()
                        .max_sets(1)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::UNIFORM_BUFFER,
                                descriptor_count: 1,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_BUFFER,
                                descriptor_count: 1,
                            },
                        ]),
                    None,
                )
                .unwrap();
            let ds = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(ds_pool)
                        .set_layouts(&[ds_layout]),
                )
                .unwrap()[0];

            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder().set_layouts(&[ds_layout]),
                    None,
                )
                .unwrap();

            let entry_point = b"main\0".as_ptr().cast();

            let pipeline = device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::builder()
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo {
                                stage: vk::ShaderStageFlags::VERTEX,
                                module: vert,
                                p_name: entry_point,
                                ..Default::default()
                            },
                            vk::PipelineShaderStageCreateInfo {
                                stage: vk::ShaderStageFlags::FRAGMENT,
                                module: frag,
                                p_name: entry_point,
                                ..Default::default()
                            },
                        ])
                        .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::builder()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::builder()
                                .scissor_count(1)
                                .viewport_count(1),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::builder()
                                .cull_mode(vk::CullModeFlags::NONE)
                                .polygon_mode(vk::PolygonMode::FILL)
                                .line_width(1.0),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::builder()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::builder()
                                .depth_test_enable(false),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::builder().attachments(&[
                                vk::PipelineColorBlendAttachmentState {
                                    blend_enable: vk::TRUE,
                                    src_color_blend_factor: vk::BlendFactor::ONE,
                                    dst_color_blend_factor: vk::BlendFactor::ONE,
                                    color_blend_op: vk::BlendOp::ADD,
                                    color_write_mask: vk::ColorComponentFlags::R
                                        | vk::ColorComponentFlags::G
                                        | vk::ColorComponentFlags::B
                                        | vk::ColorComponentFlags::A,
                                    ..Default::default()
                                },
                            ]),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(pipeline_layout)
                        .render_pass(render_pass)
                        .subpass(0)
                        .build()],
                    None,
                )
                .unwrap()[0];

            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);

            //
            // Inputs
            //
            let stars = device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size((options.star_count * mem::size_of::<StarData>()) as vk::DeviceSize)
                        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap();
            let stars_mem_reqs = device.get_buffer_memory_requirements(stars);
            let stars_mem_ty = find_memory_type(
                &memory_props,
                stars_mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .unwrap();
            let stars_mem = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .memory_type_index(stars_mem_ty)
                        .allocation_size(stars_mem_reqs.size),
                    None,
                )
                .unwrap();
            device.bind_buffer_memory(stars, stars_mem, 0).unwrap();
            let stars_ptr = device
                .map_memory(stars_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::default())
                .unwrap()
                .cast();

            let uniforms = device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(mem::size_of::<[[f32; 4]; 4]>() as vk::DeviceSize)
                        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap();
            let uniforms_mem_reqs = device.get_buffer_memory_requirements(uniforms);
            let uniforms_mem_ty = find_memory_type(
                &memory_props,
                uniforms_mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .unwrap();
            let uniforms_mem = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .memory_type_index(uniforms_mem_ty)
                        .allocation_size(uniforms_mem_reqs.size),
                    None,
                )
                .unwrap();
            device
                .bind_buffer_memory(uniforms, uniforms_mem, 0)
                .unwrap();
            let uniforms_ptr = device
                .map_memory(
                    uniforms_mem,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::default(),
                )
                .unwrap()
                .cast();

            device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::builder()
                        .dst_set(ds)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo {
                            buffer: uniforms,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        }])
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_set(ds)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo {
                            buffer: stars,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        }])
                        .build(),
                ],
                &[],
            );

            //
            // Outputs
            //

            let image = device
                .create_image(
                    &vk::ImageCreateInfo::builder()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(COLOR_FORMAT)
                        .extent(vk::Extent3D {
                            width: options.resolution,
                            height: options.resolution,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .usage(
                            vk::ImageUsageFlags::COLOR_ATTACHMENT
                                | vk::ImageUsageFlags::TRANSFER_SRC,
                        ),
                    None,
                )
                .unwrap();
            let image_mem_reqs = device.get_image_memory_requirements(image);
            let image_mem_ty = find_memory_type(
                &memory_props,
                image_mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .unwrap();
            let image_mem = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .memory_type_index(image_mem_ty)
                        .allocation_size(image_mem_reqs.size),
                    None,
                )
                .unwrap();
            device.bind_image_memory(image, image_mem, 0).unwrap();
            let image_view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                        .format(COLOR_FORMAT)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
                .unwrap();

            let transfer = device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size((options.resolution * options.resolution * 4) as vk::DeviceSize)
                        .usage(vk::BufferUsageFlags::TRANSFER_DST)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap();
            let transfer_mem_reqs = device.get_buffer_memory_requirements(transfer);
            let transfer_mem_ty = find_memory_type(
                &memory_props,
                transfer_mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED,
            )
            .unwrap();
            let transfer_mem = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .memory_type_index(transfer_mem_ty)
                        .allocation_size(transfer_mem_reqs.size),
                    None,
                )
                .unwrap();
            device
                .bind_buffer_memory(transfer, transfer_mem, 0)
                .unwrap();
            let transfer_ptr = device
                .map_memory(
                    transfer_mem,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::default(),
                )
                .unwrap()
                .cast();

            let framebuffer = device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::builder()
                        .render_pass(render_pass)
                        .attachments(&[image_view])
                        .height(options.resolution)
                        .width(options.resolution)
                        .layers(1),
                    None,
                )
                .unwrap();

            //
            // Render commands
            //

            let cmd_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::builder()
                        .queue_family_index(queue_family_index)
                        .flags(
                            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                                | vk::CommandPoolCreateFlags::TRANSIENT,
                        ),
                    None,
                )
                .unwrap();
            let cmd = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(cmd_pool)
                        .command_buffer_count(1),
                )
                .unwrap()[0];

            Self {
                _entry: entry,
                instance,
                device,
                #[cfg(feature = "renderdoc")]
                renderdoc: RenderDoc::new().ok(),
                queue,
                render_pass,

                ds_layout,
                ds_pool,
                ds,
                pipeline_layout,
                pipeline,

                resolution: options.resolution,
                star_count: options.star_count,
                stars,
                stars_mem,
                stars_ptr,
                uniforms,
                uniforms_mem,
                uniforms_ptr,

                image,
                image_mem,
                image_view,
                framebuffer,
                transfer,
                transfer_mem,
                transfer_ptr,

                cmd_pool,
                cmd,
            }
        }
    }

    fn draw(
        &mut self,
        projection: &[[f32; 4]; 4],
        stars: impl ExactSizeIterator<Item = spangle::Star>,
    ) -> Vec<u8> {
        let texels = self.resolution as usize * self.resolution as usize;
        let mut result = Vec::<u8>::with_capacity(texels * 3);
        let device = &self.device;
        let cmd = self.cmd;

        #[cfg(feature = "renderdoc")]
        if let Some(ref mut rd) = self.renderdoc {
            let rd_device = unsafe { (self.instance.handle().as_raw() as *mut *mut c_void).read() };
            rd.start_frame_capture(rd_device, std::ptr::null());
        }

        unsafe {
            let mut star_count = 0;
            for (i, star) in stars.enumerate().take(self.star_count) {
                let star = StarData::from(star);
                self.stars_ptr
                    .add(i * mem::size_of::<StarData>())
                    .copy_from_nonoverlapping(
                        (&star as *const StarData).cast(),
                        mem::size_of::<StarData>(),
                    );
                star_count += 1;
            }
            self.uniforms_ptr
                .copy_from_nonoverlapping(projection.as_ptr().cast(), mem::size_of_val(projection));

            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();

            device.cmd_begin_render_pass(
                cmd,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: vk::Extent2D {
                            width: self.resolution,
                            height: self.resolution,
                        },
                    })
                    .clear_values(&[vk::ClearValue {
                        color: vk::ClearColorValue { float32: [0.0; 4] },
                    }]),
                vk::SubpassContents::INLINE,
            );

            device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.resolution as f32,
                    height: self.resolution as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: self.resolution,
                        height: self.resolution,
                    },
                }],
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.ds],
                &[],
            );
            device.cmd_draw(cmd, star_count * 6, 1, 0, 0);

            device.cmd_end_render_pass(cmd);

            device.cmd_copy_image_to_buffer(
                cmd,
                self.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.transfer,
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: self.resolution,
                        height: self.resolution,
                        depth: 1,
                    },
                }],
            );

            device.end_command_buffer(cmd).unwrap();

            device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::builder()
                        .command_buffers(&[self.cmd])
                        .build()],
                    vk::Fence::null(),
                )
                .unwrap();
            device.queue_wait_idle(self.queue).unwrap();

            #[cfg(feature = "renderdoc")]
            if let Some(ref mut rd) = self.renderdoc {
                let rd_device = (self.instance.handle().as_raw() as *mut *mut c_void).read();
                rd.end_frame_capture(rd_device, std::ptr::null());
            }

            for offset in 0..texels {
                result
                    .as_mut_ptr()
                    .add(offset * 3)
                    .copy_from_nonoverlapping(self.transfer_ptr.add(offset * 4), 3);
            }
            result.set_len(texels * 3);
        }
        result
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let device = &self.device;
        unsafe {
            device.destroy_command_pool(self.cmd_pool, None);

            device.free_memory(self.transfer_mem, None);
            device.destroy_buffer(self.transfer, None);
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_image_view(self.image_view, None);
            device.free_memory(self.image_mem, None);
            device.destroy_image(self.image, None);

            device.free_memory(self.uniforms_mem, None);
            device.destroy_buffer(self.uniforms, None);
            device.free_memory(self.stars_mem, None);
            device.destroy_buffer(self.stars, None);

            device.destroy_descriptor_set_layout(self.ds_layout, None);
            device.destroy_descriptor_pool(self.ds_pool, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_pipeline(self.pipeline, None);

            device.destroy_render_pass(self.render_pass, None);
            device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        if type_bits & (1 << i) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(flags)
        {
            return Some(i);
        }
    }
    None
}

#[repr(C)]
#[repr(align(16))]
struct StarData {
    direction: [f32; 3],
    _pad: u32,
    irradiance: [f32; 3],
}

impl From<spangle::Star> for StarData {
    fn from(star: spangle::Star) -> Self {
        let color = spangle::black_body_color(star.temperature);
        let scale = star.irradiance / color.iter().sum::<f32>();
        Self {
            direction: star.direction,
            _pad: 0,
            irradiance: color.map(|x| x * scale),
        }
    }
}

#[rustfmt::skip]
pub fn projection(znear: f32) -> [[f32; 4]; 4] {
    // Based on http://dev.theomader.com/depth-precision/ + OpenVR docs
    let fov = std::f32::consts::FRAC_PI_4;
    let left = -fov.tan();
    let right = fov.tan();
    let down = -fov.tan();
    let up = fov.tan();
    let idx = 1.0 / (right - left);
    let idy = 1.0 / (down - up);
    let sx = right + left;
    let sy = down + up;
    [[2.0 * idx,       0.0,   0.0,  0.0],
     [      0.0, 2.0 * idy,   0.0,  0.0],
     [ sx * idx,  sy * idy,   0.0, -1.0],
     [      0.0,       0.0, znear,  0.0]]
}
