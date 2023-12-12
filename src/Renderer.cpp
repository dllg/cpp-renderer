#include "Renderer.h"

#include <yocto/yocto_sceneio.h>
#include <yocto/yocto_image.h>
#include <yocto/yocto_gui.h>
#include <yocto/yocto_math.h>

#include "yocto_gui_snapshot.h"

bool Renderer::loadGltf(const std::string &filename)
{
    if (!yocto::load_scene(filename, _scene, _err, false))
    {
        return false;
    }
    return true;
}

// Add a camera facing angle degrees towards the scene
void Renderer::addCamera(float angleDegrees)
{
    _scene.camera_names.emplace_back("camera");
    auto &camera = _scene.cameras.emplace_back();
    camera.orthographic = false;
    camera.film = 0.036f;
    camera.aspect = (float)4 / (float)3;
    camera.aperture = 0;
    camera.lens = 0.050f;
    auto bbox = compute_bounds(_scene);
    auto center = (bbox.max + bbox.min) / 2;
    auto bbox_radius = length(bbox.max - bbox.min) / 2;

    auto camera_dir = yocto::vec3f{0, 0, 1};

    // Define the rotation angle in radians
    float angle = angleDegrees * M_PI / 180.0f;

    // Create a rotation matrix for a 45 degrees rotation around the X-axis
    yocto::mat3f rotation_matrix = {
        {1, 0, 0},
        {0, cosf(angle), -sinf(angle)},
        {0, sinf(angle), cosf(angle)}};

    // Multiply the camera direction vector by the rotation matrix
    camera_dir = rotation_matrix * camera_dir;
    auto camera_dist = bbox_radius * camera.lens / (camera.film / camera.aspect);
    camera_dist *= 2.0f; // correction for tracer camera implementation
    auto from = camera_dir * camera_dist + center;
    auto to = center;
    auto up = yocto::vec3f{0, 1, 0};
    camera.frame = lookat_frame(from, to, up);
    camera.focus = length(from - to);
}

bool Renderer::render(const std::string &filename, int width, int height)
{
    if (_scene.shapes.empty())
    {
        _err = "No shape in scene";
        return false;
    }
    addCamera(45.0f); // Add a camera facing 45 degrees towards the scene
    auto params = yocto::shade_params{};
    params.camera = static_cast<int>(_scene.cameras.size()) - 1;
    params.exposure = 1.5f;
    params.gamma = 4.0f;
    save_shade_gui_snapshot(_scene,
                            params,
                            yocto::vec2i{width, height},
                            filename);

    return true;
}
