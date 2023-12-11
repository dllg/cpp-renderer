#pragma once

#include <yocto/yocto_scene.h>

#include <tiny_obj_loader.h>

#include <string>
#include <vector>


class Renderer
{
public:
    Renderer() {}
    bool loadGltf(const std::string &filename);
    bool render(const std::string &filename, int width, int height);
    const std::string &getError() const { return _err; }

private:
    yocto::scene_data _scene;
    std::string _err;
};
