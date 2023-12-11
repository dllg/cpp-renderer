#include "Renderer.h"

#include <argh.h>
#include <spdlog/spdlog.h>

int main(int argc, char **argv)
{
    argh::parser cmdl;
    cmdl.parse(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);

    if (cmdl.size() < 2)
    {
        printf("Usage: %s <input file (.gltf)> <output file (.png)>\n", argv[0]);
        return 1;
    }

    std::string input = cmdl[1];
    std::string output = cmdl[2];

    Renderer renderer;
    if (!renderer.loadGltf(input))
    {
        spdlog::error("Error: {}", renderer.getError());
        return 1;
    }
    if (!renderer.render(output, 640, 480))
    {
        spdlog::error("Error: {}", renderer.getError());
        return 1;
    }
    return 0;
}
