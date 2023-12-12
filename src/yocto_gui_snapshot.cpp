//
// Simpler image viewer.
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2022 Fabio Pellacini
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//

#ifdef YOCTO_OPENGL

#include <glad/glad.h>

#include <cassert>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "yocto/yocto_cutrace.h"
#include "yocto/yocto_geometry.h"
#include "yocto/yocto_gui.h"
#include "yocto/yocto_scene.h"
#include "yocto/yocto_sceneio.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <imgui/imgui.h>
#include <imgui_internal.h>

#ifdef _WIN32
#undef near
#undef far
#endif

// -----------------------------------------------------------------------------
// SCENE DRAWING
// -----------------------------------------------------------------------------
namespace yocto
{

    // Opengl texture
    struct glscene_texture
    {
        // texture properties
        int width = 0;
        int height = 0;

        // opengl state
        uint texture = 0;
    };

    // Opengl shape
    struct glscene_shape
    {
        // Shape properties
        int num_positions = 0;
        int num_normals = 0;
        int num_texcoords = 0;
        int num_colors = 0;
        int num_tangents = 0;
        int num_points = 0;
        int num_lines = 0;
        int num_triangles = 0;
        int num_quads = 0;

        // OpenGl state
        uint vertexarray = 0;
        uint positions = 0;
        uint normals = 0;
        uint texcoords = 0;
        uint colors = 0;
        uint tangents = 0;
        uint points = 0;
        uint lines = 0;
        uint triangles = 0;
        uint quads = 0;
        float point_size = 1;
    };

    // Opengl scene
    struct glscene_state
    {
        // scene objects
        vector<glscene_shape> shapes = {};
        vector<glscene_texture> textures = {};

        // programs
        uint program = 0;
        uint vertex = 0;
        uint fragment = 0;
    };

} // namespace yocto

// -----------------------------------------------------------------------------
// WINDOW
// -----------------------------------------------------------------------------
namespace yocto
{

    // OpenGL window wrapper
    struct glwindow_state
    {
        string title = "";
        gui_callback init = {};
        gui_callback clear = {};
        gui_callback draw = {};
        gui_callback widgets = {};
        gui_callback update = {};
        gui_callback uiupdate = {};
        int widgets_width = 0;
        bool widgets_left = true;
        gui_input input = {};
        vec2i window = {0, 0};
        vec4f background = {0.15f, 0.15f, 0.15f, 1.0f};
    };

    static void draw_window(glwindow_state &state)
    {
        glClearColor(state.background.x, state.background.y, state.background.z,
                     state.background.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (state.draw)
            state.draw(state.input);
        if (state.widgets)
        {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            auto window = state.window;
            if (state.widgets_left)
            {
                ImGui::SetNextWindowPos({0, 0});
                ImGui::SetNextWindowSize({(float)state.widgets_width, (float)window.y});
            }
            else
            {
                ImGui::SetNextWindowPos({(float)(window.x - state.widgets_width), 0});
                ImGui::SetNextWindowSize({(float)state.widgets_width, (float)window.y});
            }
            ImGui::SetNextWindowCollapsed(false);
            ImGui::SetNextWindowBgAlpha(1);
            if (ImGui::Begin(state.title.c_str(), nullptr,
                             ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                                 ImGuiWindowFlags_NoSavedSettings))
            {
                state.widgets(state.input);
            }
            ImGui::End();
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }
    }
} // namespace yocto

// -----------------------------------------------------------------------------
// OPENGL HELPERS
// -----------------------------------------------------------------------------
namespace yocto
{

    // assert on error
    [[maybe_unused]] static GLenum _assert_ogl_error()
    {
        auto error_code = glGetError();
        if (error_code != GL_NO_ERROR)
        {
            auto error = string{};
            switch (error_code)
            {
            case GL_INVALID_ENUM:
                error = "INVALID_ENUM";
                break;
            case GL_INVALID_VALUE:
                error = "INVALID_VALUE";
                break;
            case GL_INVALID_OPERATION:
                error = "INVALID_OPERATION";
                break;
            // case GL_STACK_OVERFLOW: error = "STACK_OVERFLOW"; break;
            // case GL_STACK_UNDERFLOW: error = "STACK_UNDERFLOW"; break;
            case GL_OUT_OF_MEMORY:
                error = "OUT_OF_MEMORY";
                break;
            case GL_INVALID_FRAMEBUFFER_OPERATION:
                error = "INVALID_FRAMEBUFFER_OPERATION";
                break;
            }
            printf("\n    OPENGL ERROR: %s\n\n", error.c_str());
        }
        return error_code;
    }
    static void assert_glerror() { assert(_assert_ogl_error() == GL_NO_ERROR); }

    // initialize program
    static void set_program(uint &program_id, uint &vertex_id, uint &fragment_id,
                            const string &vertex, const string &fragment)
    {
        // error
        auto program_error = [&](const char *message, const char *log)
        {
            if (program_id)
                glDeleteProgram(program_id);
            if (vertex_id)
                glDeleteShader(program_id);
            if (fragment_id)
                glDeleteShader(program_id);
            program_id = 0;
            vertex_id = 0;
            fragment_id = 0;
            printf("%s\n", message);
            printf("%s\n", log);
        };

        const char *ccvertex = vertex.data();
        const char *ccfragment = fragment.data();
        auto errflags = 0;
        auto errbuf = array<char, 10000>{};

        assert_glerror();

        // create vertex
        vertex_id = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_id, 1, &ccvertex, NULL);
        glCompileShader(vertex_id);
        glGetShaderiv(vertex_id, GL_COMPILE_STATUS, &errflags);
        if (errflags == 0)
        {
            glGetShaderInfoLog(vertex_id, 10000, 0, errbuf.data());
            return program_error("vertex shader not compiled", errbuf.data());
        }
        assert_glerror();

        // create fragment
        fragment_id = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_id, 1, &ccfragment, NULL);
        glCompileShader(fragment_id);
        glGetShaderiv(fragment_id, GL_COMPILE_STATUS, &errflags);
        if (errflags == 0)
        {
            glGetShaderInfoLog(fragment_id, 10000, 0, errbuf.data());
            return program_error("fragment shader not compiled", errbuf.data());
        }
        assert_glerror();

        // create program
        program_id = glCreateProgram();
        glAttachShader(program_id, vertex_id);
        glAttachShader(program_id, fragment_id);
        glLinkProgram(program_id);
        glGetProgramiv(program_id, GL_LINK_STATUS, &errflags);
        if (errflags == 0)
        {
            glGetProgramInfoLog(program_id, 10000, 0, errbuf.data());
            return program_error("program not linked", errbuf.data());
        }
// TODO(fabio): Apparently validation must be done just before drawing.
//    https://community.khronos.org/t/samplers-of-different-types-use-the-same-textur/66329
// If done here, validation fails when using cubemaps and textures in the
// same shader. We should create a function validate_program() anc call it
// separately.
#if 0
  glValidateProgram(program_id);
  glGetProgramiv(program_id, GL_VALIDATE_STATUS, &errflags);
  if (!errflags) {
    glGetProgramInfoLog(program_id, 10000, 0, errbuf.data());
    return program_error("program not validated", errbuf.data());
  }
  assert_glerror();
#endif
    }

} // namespace yocto

// -----------------------------------------------------------------------------
// SCENE DRAWING
// -----------------------------------------------------------------------------
namespace yocto
{

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

    static const char *glscene_vertex =
        R"(
#version 330

layout(location = 0) in vec3 positions;           // vertex position (in mesh coordinate frame)
layout(location = 1) in vec3 normals;             // vertex normal (in mesh coordinate frame)
layout(location = 2) in vec2 texcoords;           // vertex texcoords
layout(location = 3) in vec4 colors;              // vertex color
layout(location = 4) in vec4 tangents;            // vertex tangent space

uniform mat4 frame;             // shape transform
uniform mat4 frameit;           // shape transform

uniform mat4 view;              // inverse of the camera frame (as a matrix)
uniform mat4 projection;        // camera projection

out vec3 position;              // [to fragment shader] vertex position (in world coordinate)
out vec3 normal;                // [to fragment shader] vertex normal (in world coordinate)
out vec2 texcoord;              // [to fragment shader] vertex texture coordinates
out vec4 scolor;                // [to fragment shader] vertex color
out vec4 tangsp;                // [to fragment shader] vertex tangent space

// main function
void main() {
  // copy values
  position = positions;
  normal = normals;
  tangsp = tangents;
  texcoord = texcoords;
  scolor = colors;

  // world projection
  position = (frame * vec4(position,1)).xyz;
  normal = (frameit * vec4(normal,0)).xyz;
  tangsp.xyz = (frame * vec4(tangsp.xyz,0)).xyz;

  // clip
  gl_Position = projection * view * vec4(position,1);
}
)";

    static const char *glscene_fragment =
        R"(
#version 330

in vec3 position;  // [from vertex shader] position in world space
in vec3 normal;    // [from vertex shader] normal in world space
in vec2 texcoord;  // [from vertex shader] texcoord
in vec4 scolor;    // [from vertex shader] color
in vec4 tangsp;    // [from vertex shader] tangent space

uniform int element;
uniform bool unlit;
uniform bool faceted;
uniform vec4 highlight;
uniform bool double_sided;

uniform vec3 emission;            // material ke
uniform vec3 color;               // material kd
uniform float specular;           // material ks
uniform float metallic;           // material km
uniform float roughness;          // material rs
uniform float opacity;            // material op

uniform bool emission_tex_on;     // material ke texture on
uniform sampler2D emission_tex;   // material ke texture
uniform bool color_tex_on;        // material kd texture on
uniform sampler2D color_tex;      // material kd texture
uniform bool roughness_tex_on;    // material rs texture on
uniform sampler2D roughness_tex;  // material rs texture
uniform bool normalmap_tex_on;    // material normal texture on
uniform sampler2D normalmap_tex;  // material normal texture

uniform int  lighting;            // eyelight shading
uniform vec3 ambient;             // ambient light
uniform int  lights_num;          // number of lights
uniform vec3 lights_direction[16];// light positions
uniform vec3 lights_emission[16]; // light intensities

uniform mat4 frame;              // shape transform
uniform mat4 frameit;            // shape transform

uniform vec3 eye;              // camera position
uniform mat4 view;             // inverse of the camera frame (as a matrix)
uniform mat4 projection;       // camera projection

uniform float exposure;
uniform float gamma;

out vec4 frag_color;

float pif = 3.14159265;

struct shade_brdf {
  vec3  emission;
  vec3  diffuse;
  vec3  specular;
  float roughness;
  float opacity;
};

vec3 eval_brdf_color(vec3 value, sampler2D tex, bool tex_on) {
  vec3 result = value;
  if (tex_on) result *= texture(tex, texcoord).rgb;
  return result;
}
float eval_brdf_value(float value, sampler2D tex, bool tex_on) {
  float result = value;
  if (tex_on) result *= texture(tex, texcoord).r;
  return result;
}

shade_brdf eval_brdf() {
  vec4 emission_t = vec4(emission, 1);
  if (emission_tex_on) emission_t *= texture(emission_tex, texcoord);
  vec4 base_t = scolor * vec4(color, opacity);
  if (color_tex_on) base_t *= pow(texture(color_tex, texcoord), vec4(2.2,2.2,2.2,1));
  float metallic_t = metallic;
  float roughness_t = roughness;
  roughness_t = roughness_t * roughness_t;
  if (roughness_t < 0.03 * 0.03) roughness_t = 0.03 * 0.03;
  float specular_t = specular;

  // color?
  shade_brdf brdf;
  brdf.emission  = emission_t.xyz;
  brdf.diffuse   = base_t.xyz * (1 - metallic_t);
  brdf.specular  = specular_t * (base_t.xyz * metallic_t + vec3(0.04) * (1 - metallic_t));
  brdf.roughness = roughness_t;
  brdf.opacity   = base_t.w;
  return brdf;
}

vec3 eval_brdfcos(shade_brdf brdf, vec3 n, vec3 incoming, vec3 outgoing) {
  vec3 halfway = normalize(incoming+outgoing);
  float ndi = dot(incoming,n), ndo = dot(outgoing,n), ndh = dot(halfway,n);
  if(ndi<=0 || ndo <=0) return vec3(0);
  vec3 diff = ndi * brdf.diffuse / pif;
  if(ndh<=0) return diff;
  float cos2 = ndh * ndh;
  float tan2 = (1 - cos2) / cos2;
  float alpha2 = brdf.roughness * brdf.roughness;
  float d = alpha2 / (pif * cos2 * cos2 * (alpha2 + tan2) * (alpha2 + tan2));
  float lambda_o = (-1 + sqrt(1 + (1 - ndo * ndo) / (ndo * ndo))) / 2;
  float lambda_i = (-1 + sqrt(1 + (1 - ndi * ndi) / (ndi * ndi))) / 2;
  float g = 1 / (1 + lambda_o + lambda_i);
  vec3 spec = ndi * brdf.specular * d * g / (4*ndi*ndo);
  return diff+spec;
}

vec3 apply_normal_map(vec2 texcoord, vec3 normal, vec4 tangsp) {
  if(!normalmap_tex_on) return normal;
  vec3 tangu = normalize((frame * vec4(normalize(tangsp.xyz),0)).xyz);
  vec3 tangv = normalize(cross(normal, tangu));
  if(tangsp.w < 0) tangv = -tangv;
  vec3 texture = 2 * texture(normalmap_tex,texcoord).xyz - 1;
  // texture.y = -texture.y;
  return normalize( tangu * texture.x + tangv * texture.y + normal * texture.z );
}

vec3 triangle_normal(vec3 position) {
  vec3 fdx = dFdx(position);
  vec3 fdy = dFdy(position);
  return normalize((frame * vec4(normalize(cross(fdx, fdy)), 0)).xyz);
}

#define element_points 1
#define element_lines 2
#define element_triangles 3

vec3 eval_normal(vec3 outgoing) {
  vec3 norm;
  if (element == element_triangles) {
    if (faceted) {
      norm = triangle_normal(position);
    } else {
      norm = normalize(normal);
    }
    // apply normal map
    norm = apply_normal_map(texcoord, norm, tangsp);
    if (double_sided) norm = faceforward(norm, -outgoing, norm);
  }

  if (element == element_lines) {
    vec3 tangent = normalize(normal);
    norm         = normalize(outgoing - tangent * dot(outgoing, tangent));
  }

  return norm;
}

#define lighting_eyelight 0
#define lighting_camlight 1

// main
void main() {
  // view vector
  vec3 outgoing = normalize(eye - position);
  vec3 n = eval_normal(outgoing);

  // get material color from textures
  shade_brdf brdf = eval_brdf();
  if(brdf.opacity < 0.005) discard;

  if(unlit) {
    frag_color = vec4(brdf.emission + brdf.diffuse, brdf.opacity);
    return;
  }

  // emission
  vec3 radiance = brdf.emission;

  // check early exit
  if(brdf.diffuse != vec3(0,0,0) || brdf.specular != vec3(0,0,0)) {
    // eyelight shading
    if(lighting == lighting_eyelight) {
      vec3 incoming = outgoing;
      radiance += pif * eval_brdfcos(brdf, n, incoming, outgoing);
    }
    if(lighting == lighting_camlight) {
      // accumulate ambient
      radiance += ambient * brdf.diffuse;
      // foreach light
      for(int lid = 0; lid < lights_num; lid ++) {
        radiance += lights_emission[lid] *
          eval_brdfcos(brdf, n, lights_direction[lid], outgoing);
      }
    }
  }

  // final color correction
  radiance = pow(radiance * pow(2,exposure), vec3(1/gamma));

  // highlighting
  if(highlight.w > 0) {
    if(mod(int(gl_FragCoord.x)/4 + int(gl_FragCoord.y)/4, 2)  == 0)
        radiance = highlight.xyz * highlight.w + radiance * (1-highlight.w);
  }

  // output final color by setting gl_FragColor
  frag_color = vec4(radiance, brdf.opacity);
}
)";

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif

    // Create texture
    static void set_texture(
        glscene_texture &gltexture, const texture_data &texture)
    {
        if (!gltexture.texture || gltexture.width != texture.width ||
            gltexture.height != texture.height)
        {
            if (!gltexture.texture)
                glGenTextures(1, &gltexture.texture);
            glBindTexture(GL_TEXTURE_2D, gltexture.texture);
            if (!texture.pixelsb.empty())
            {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.width, texture.height, 0,
                             GL_RGBA, GL_UNSIGNED_BYTE, texture.pixelsb.data());
            }
            else
            {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.width, texture.height, 0,
                             GL_RGBA, GL_FLOAT, texture.pixelsf.data());
            }
            glGenerateMipmap(GL_TEXTURE_2D);
            glTexParameteri(
                GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }
        else
        {
            glBindTexture(GL_TEXTURE_2D, gltexture.texture);
            if (!texture.pixelsb.empty())
            {
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture.width, texture.height,
                                GL_RGBA, GL_UNSIGNED_BYTE, texture.pixelsb.data());
            }
            else
            {
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture.width, texture.height,
                                GL_RGBA, GL_FLOAT, texture.pixelsf.data());
            }
            glGenerateMipmap(GL_TEXTURE_2D);
        }
    }

    // Create shape
    static void set_shape(glscene_shape &glshape, const shape_data &shape)
    {
        auto set_vertex = [](uint &buffer, int &num, const auto &data,
                             const auto &def, int location)
        {
            if (data.empty())
            {
                if (buffer)
                    glDeleteBuffers(1, &buffer);
                buffer = 0;
                num = 0;
                glDisableVertexAttribArray(location);
                if constexpr (sizeof(def) == sizeof(float))
                    glVertexAttrib1f(location, (float)def);
                if constexpr (sizeof(def) == sizeof(vec2f))
                    glVertexAttrib2fv(location, (float *)&def.x);
                if constexpr (sizeof(def) == sizeof(vec3f))
                    glVertexAttrib3fv(location, (float *)&def.x);
                if constexpr (sizeof(def) == sizeof(vec4f))
                    glVertexAttrib4fv(location, (float *)&def.x);
            }
            else
            {
                if (!buffer || (int)data.size() != num)
                {
                    if (buffer)
                        glDeleteBuffers(1, &buffer);
                    glGenBuffers(1, &buffer);
                    glBindBuffer(GL_ARRAY_BUFFER, buffer);
                    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(data.front()),
                                 data.data(), GL_STATIC_DRAW);
                    num = (int)data.size();
                }
                else
                {
                    // we have enough space
                    glBindBuffer(GL_ARRAY_BUFFER, buffer);
                    glBufferSubData(GL_ARRAY_BUFFER, 0, data.size() * sizeof(data.front()),
                                    data.data());
                }
                glBindBuffer(GL_ARRAY_BUFFER, buffer);
                glEnableVertexAttribArray(location);
                glVertexAttribPointer(location, sizeof(data.front()) / sizeof(float),
                                      GL_FLOAT, false, 0, nullptr);
            }
        };

        auto set_indices = [](uint &buffer, int &num, const auto &data)
        {
            if (data.empty())
            {
                if (buffer)
                    glDeleteBuffers(1, &buffer);
                buffer = 0;
                num = 0;
            }
            else
            {
                if (!buffer || (int)data.size() != num)
                {
                    if (buffer)
                        glDeleteBuffers(1, &buffer);
                    glGenBuffers(1, &buffer);
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
                    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                                 data.size() * sizeof(data.front()), data.data(), GL_STATIC_DRAW);
                    num = (int)data.size();
                }
                else
                {
                    // we have enough space
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
                    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0,
                                    data.size() * sizeof(data.front()), data.data());
                }
            }
        };

        if (!glshape.vertexarray)
            glGenVertexArrays(1, &glshape.vertexarray);
        glBindVertexArray(glshape.vertexarray);
        set_indices(glshape.points, glshape.num_points, shape.points);
        set_indices(glshape.lines, glshape.num_lines, shape.lines);
        set_indices(glshape.triangles, glshape.num_triangles, shape.triangles);
        set_indices(
            glshape.quads, glshape.num_quads, quads_to_triangles(shape.quads));
        set_vertex(glshape.positions, glshape.num_positions, shape.positions,
                   vec3f{0, 0, 0}, 0);
        set_vertex(
            glshape.normals, glshape.num_normals, shape.normals, vec3f{0, 0, 1}, 1);
        set_vertex(glshape.texcoords, glshape.num_texcoords, shape.texcoords,
                   vec2f{0, 0}, 2);
        set_vertex(
            glshape.colors, glshape.num_colors, shape.colors, vec4f{1, 1, 1, 1}, 3);
        set_vertex(glshape.tangents, glshape.num_tangents, shape.tangents,
                   vec4f{0, 0, 1, 1}, 4);
        glBindVertexArray(0);
    }

    // init scene
    static void init_glscene(glscene_state &glscene, const scene_data &ioscene)
    {
        // program
        set_program(glscene.program, glscene.vertex, glscene.fragment, glscene_vertex,
                    glscene_fragment);

        // textures
        for (auto &iotexture : ioscene.textures)
        {
            auto &gltexture = glscene.textures.emplace_back();
            set_texture(gltexture, iotexture);
        }

        // shapes
        for (auto &ioshape : ioscene.shapes)
        {
            auto &glshape = glscene.shapes.emplace_back();
            set_shape(glshape, ioshape);
        }
    }

    static void draw_scene(glscene_state &glscene, const scene_data &scene,
                           const vec4i &viewport, const shade_params &params)
    {
        // check errors
        assert_glerror();

        // viewport and framebuffer
        glViewport(viewport.x, viewport.y, viewport.z, viewport.w);
        glClearColor(params.background.x, params.background.y, params.background.z,
                     params.background.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        // set program
        auto &program = glscene.program;
        glUseProgram(program);

        // camera
        auto &camera = scene.cameras.at(params.camera);
        auto camera_aspect = (float)viewport.z / (float)viewport.w;
        auto camera_yfov =
            camera_aspect >= 0
                ? (2 * atan(camera.film / (camera_aspect * 2 * camera.lens)))
                : (2 * atan(camera.film / (2 * camera.lens)));
        auto view_matrix = frame_to_mat(inverse(camera.frame));
        auto projection_matrix = perspective_mat(
            camera_yfov, camera_aspect, params.near, params.far);
        glUniform3f(glGetUniformLocation(program, "eye"), camera.frame.o.x,
                    camera.frame.o.y, camera.frame.o.z);
        glUniformMatrix4fv(
            glGetUniformLocation(program, "view"), 1, false, &view_matrix.x.x);
        glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1, false,
                           &projection_matrix.x.x);

        // params
        glUniform1f(glGetUniformLocation(program, "exposure"), params.exposure);
        glUniform1f(glGetUniformLocation(program, "gamma"), params.gamma);
        glUniform1i(glGetUniformLocation(program, "double_sided"),
                    params.double_sided ? 1 : 0);

        static auto lights_direction = vector<vec3f>{normalize(vec3f{1, 1, 1}),
                                                     normalize(vec3f{-1, 1, 1}), normalize(vec3f{-1, -1, 1}),
                                                     normalize(vec3f{0.1f, 0.5f, -1})};
        static auto lights_emission = vector<vec3f>{vec3f{pif / 2, pif / 2, pif / 2},
                                                    vec3f{pif / 2, pif / 2, pif / 2}, vec3f{pif / 4, pif / 4, pif / 4},
                                                    vec3f{pif / 4, pif / 4, pif / 4}};
        if (params.lighting == shade_lighting::camlight)
        {
            glUniform1i(glGetUniformLocation(program, "lighting"), 1);
            glUniform3f(glGetUniformLocation(program, "ambient"), 0, 0, 0);
            glUniform1i(glGetUniformLocation(program, "lights_num"),
                        (int)lights_direction.size());
            for (auto lid : range((int)lights_direction.size()))
            {
                auto is = std::to_string(lid);
                auto direction = transform_direction(camera.frame, lights_direction[lid]);
                glUniform3f(glGetUniformLocation(
                                program, ("lights_direction[" + is + "]").c_str()),
                            direction.x, direction.y, direction.z);
                glUniform3f(glGetUniformLocation(
                                program, ("lights_emission[" + is + "]").c_str()),
                            lights_emission[lid].x, lights_emission[lid].y,
                            lights_emission[lid].z);
            }
        }
        else if (params.lighting == shade_lighting::eyelight)
        {
            glUniform1i(glGetUniformLocation(program, "lighting"), 0);
            glUniform1i(glGetUniformLocation(program, "lights_num"), 0);
        }
        else
        {
            throw std::invalid_argument{"unknown lighting type"};
        }

        // helper
        auto set_texture = [&glscene](uint program, const char *name,
                                      const char *name_on, int texture_idx, int unit)
        {
            if (texture_idx >= 0)
            {
                auto &gltexture = glscene.textures.at(texture_idx);
                glActiveTexture(GL_TEXTURE0 + unit);
                glBindTexture(GL_TEXTURE_2D, gltexture.texture);
                glUniform1i(glGetUniformLocation(program, name), unit);
                glUniform1i(glGetUniformLocation(program, name_on), 1);
            }
            else
            {
                glActiveTexture(GL_TEXTURE0 + unit);
                glBindTexture(GL_TEXTURE_2D, 0);
                glUniform1i(glGetUniformLocation(program, name), unit);
                glUniform1i(glGetUniformLocation(program, name_on), 0);
            }
        };

        // draw instances
        if (params.wireframe)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        for (auto &instance : scene.instances)
        {
            auto &glshape = glscene.shapes.at(instance.shape);
            auto &material = scene.materials.at(instance.material);

            auto shape_xform = frame_to_mat(instance.frame);
            auto shape_inv_xform = transpose(
                frame_to_mat(inverse(instance.frame, params.non_rigid_frames)));
            glUniformMatrix4fv(
                glGetUniformLocation(program, "frame"), 1, false, &shape_xform.x.x);
            glUniformMatrix4fv(glGetUniformLocation(program, "frameit"), 1, false,
                               &shape_inv_xform.x.x);
            glUniform1i(glGetUniformLocation(program, "faceted"),
                        (params.faceted || glshape.normals == 0) ? 1 : 0);

            glUniform1i(glGetUniformLocation(program, "unlit"), 0);
            glUniform3f(glGetUniformLocation(program, "emission"), material.emission.x,
                        material.emission.y, material.emission.z);
            glUniform3f(glGetUniformLocation(program, "color"), material.color.x,
                        material.color.y, material.color.z);
            glUniform1f(glGetUniformLocation(program, "specular"), 1);
            glUniform1f(glGetUniformLocation(program, "metallic"), material.metallic);
            glUniform1f(glGetUniformLocation(program, "roughness"), material.roughness);
            glUniform1f(glGetUniformLocation(program, "opacity"), material.opacity);
            if (material.type == material_type::matte ||
                material.type == material_type::transparent ||
                material.type == material_type::refractive ||
                material.type == material_type::subsurface ||
                material.type == material_type::volumetric)
            {
                glUniform1f(glGetUniformLocation(program, "specular"), 0);
            }
            if (material.type == material_type::reflective)
            {
                glUniform1f(glGetUniformLocation(program, "metallic"), 1);
            }
            glUniform1i(glGetUniformLocation(program, "double_sided"),
                        params.double_sided ? 1 : 0);
            set_texture(
                program, "emission_tex", "emission_tex_on", material.emission_tex, 0);
            set_texture(program, "color_tex", "color_tex_on", material.color_tex, 1);
            set_texture(program, "roughness_tex", "roughness_tex_on",
                        material.roughness_tex, 3);
            set_texture(
                program, "normalmap_tex", "normalmap_tex_on", material.normal_tex, 5);
            assert_glerror();

            if (glshape.points)
                glUniform1i(glGetUniformLocation(program, "element"), 1);
            if (glshape.lines)
                glUniform1i(glGetUniformLocation(program, "element"), 2);
            if (glshape.triangles)
                glUniform1i(glGetUniformLocation(program, "element"), 3);
            if (glshape.quads)
                glUniform1i(glGetUniformLocation(program, "element"), 3);
            assert_glerror();

            glBindVertexArray(glshape.vertexarray);
            if (glshape.points)
            {
                glPointSize(glshape.point_size);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glshape.points);
                glDrawElements(
                    GL_POINTS, (GLsizei)glshape.num_points * 1, GL_UNSIGNED_INT, nullptr);
                glPointSize(glshape.point_size);
            }
            if (glshape.lines)
            {
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glshape.lines);
                glDrawElements(
                    GL_LINES, (GLsizei)glshape.num_lines * 2, GL_UNSIGNED_INT, nullptr);
            }
            if (glshape.triangles)
            {
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glshape.triangles);
                glDrawElements(GL_TRIANGLES, (GLsizei)glshape.num_triangles * 3,
                               GL_UNSIGNED_INT, nullptr);
            }
            if (glshape.quads)
            {
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glshape.quads);
                glDrawElements(GL_TRIANGLES, (GLsizei)glshape.num_quads * 3,
                               GL_UNSIGNED_INT, nullptr);
            }

            glBindVertexArray(0);
            assert_glerror();
        }
        if (params.wireframe)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // done
        glUseProgram(0);
    }

} // namespace yocto

namespace yocto
{
    void draw_and_get_buffer(const vec2i &size, const string &title, const gui_callbacks &callbacks, std::vector<float> &buffer)
    {
        // init glfw
        if (!glfwInit())
            throw std::runtime_error("cannot initialize windowing system");
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#if __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

        // create state
        auto state = glwindow_state{};
        state.title = title;
        state.init = callbacks.init;
        state.draw = callbacks.draw;
        state.update = callbacks.update;

        // create window
        auto window = glfwCreateWindow(
            size.x, size.y, title.c_str(), nullptr, nullptr);
        if (window == nullptr)
            throw std::runtime_error{"cannot initialize windowing system"};
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1); // Enable vsync

        // set user data
        glfwSetWindowUserPointer(window, &state);

        // set callbacks
        glfwSetWindowRefreshCallback(window, [](GLFWwindow *window)
                                     {
            auto& state = *(glwindow_state*)glfwGetWindowUserPointer(window);
            glfwGetWindowSize(window, &state.window.x, &state.window.y);
            draw_window(state);
            glfwSwapBuffers(window); });

        // init gl extensions
        if (!gladLoadGL())
            throw std::runtime_error{"cannot initialize OpenGL extensions"};

        // init
        if (state.init)
            state.init(state.input);

        glfwGetWindowSize(window, &state.input.window.x, &state.input.window.y);
        glfwGetFramebufferSize(
            window, &state.input.framebuffer.z, &state.input.framebuffer.w);
        state.input.framebuffer.x = 0;
        state.input.framebuffer.y = 0;

        // update
        if (state.update)
            state.update(state.input);

        // draw
        glfwGetWindowSize(window, &state.window.x, &state.window.y);
        draw_window(state);

        // get buffer
        glReadPixels(0, 0, size.x, size.y, GL_RGB, GL_FLOAT, buffer.data());

        // clear
        glfwDestroyWindow(window);
        glfwTerminate();
    }
} // namespace yocto

void save_shade_gui_snapshot(yocto::scene_data &scene,
                             const yocto::shade_params &params,
                             const yocto::vec2i &size,
                             const std::string &filename)
{
    // glscene
    auto glscene = yocto::glscene_state{};

    // callbacks
    auto callbacks = yocto::gui_callbacks{};
    callbacks.init = [&](const yocto::gui_input &input)
    {
        yocto::init_glscene(glscene, scene);
    };
    callbacks.draw = [&](const yocto::gui_input &input)
    {
        yocto::draw_scene(glscene, scene, input.framebuffer, params);
    };

    std::vector<float> buffer(size.x * size.y * 3);
    yocto::draw_and_get_buffer(size, "testing", callbacks, buffer);

    auto image = yocto::make_image(size.x, size.y, true);
    for (auto j = 0; j < image.height; j++)
    {
        for (auto i = 0; i < image.width; i++)
        {
            yocto::set_pixel(image, i, j, {buffer[(j * image.width + i) * 3 + 0], buffer[(j * image.width + i) * 3 + 1], buffer[(j * image.width + i) * 3 + 2], 1.0f});
        }
    }
    yocto::save_image(filename, image);
}

#endif // YOCTO_OPENGL