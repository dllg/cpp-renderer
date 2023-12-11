#include "Renderer.h"

#include <yocto/yocto_sceneio.h>

bool Renderer::loadGltf(const std::string &filename)
{
    if (!yocto::load_scene(filename, _scene, _err, false))
    {
        return false;
    }
    return true;
}

bool Renderer::render(const std::string &filename, int width, int height)
{
    (void)filename;
    (void)width;
    (void)height;
    return true;
}
