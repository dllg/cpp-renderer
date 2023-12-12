#pragma once

#include <yocto/yocto_scene.h>
#include <yocto/yocto_math.h>

#include <string>

#ifdef YOCTO_OPENGL
void save_shade_gui_snapshot(yocto::scene_data &scene,
                             const yocto::shade_params &params,
                             const yocto::vec2i &size,
                             const std::string &filename);
#endif // YOCTO_OPENGL