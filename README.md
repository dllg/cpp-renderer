# cpp-renderer
A renderer in C++ to render a gltf file as a png snapshot. Uses [yocto-gl](https://github.com/xelatihy/yocto-gl) for loading and rendering. Just adds a simple layer for creating a snapshot picture using opengl.

## Building

Install [vcpkg](https://vcpkg.io) and packages

```
$ git clone https://git@github.com/microsoft/vcpkg
$ cd vcpkg
$ ./bootstrap-vcpkg.sh
$ ./vcpkg install argh spdlog
```

Build with [cmake](http://cmake.org) and [ninja](https://ninja-build.org/)
```
$ mkdir build
$ cmake .. -G Ninja
$ ninja -j 10
```

Install [Xvfb](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml)
```
$ sudo apt update
$ sudo apt install xvfb
```

## Running

Run Xvfb
```
$ . ./start_xvfb.sh
```

Run the renderer
```
$ ./build/bin/renderer test.gltf test.png
```
