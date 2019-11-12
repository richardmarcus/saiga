﻿/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// I'm not sure why, but this pragma once doesn't work with precompiled headers and GCC 8/9.
// TODO: Check after a new gcc version comes out.
//#pragma once
#ifndef SAIGA_OPENGL_OPENGL_H
#define SAIGA_OPENGL_OPENGL_H

#include "saiga/config.h"

#ifndef SAIGA_USE_OPENGL
#    error Saiga was build without opengl.
#endif

#include <glbinding/ProcAddress.h>
#include <glbinding/gl/gl.h>
#include <vector>
// make sure nobody else includes gl.h after this
#define __gl_h_
using namespace gl;
#define GLFW_INCLUDE_NONE



namespace Saiga
{
SAIGA_OPENGL_API std::ostream& operator<<(std::ostream& os, GLenum g);

SAIGA_OPENGL_API void initOpenGL(glbinding::GetProcAddress func);
SAIGA_OPENGL_API void terminateOpenGL();
SAIGA_OPENGL_API bool OpenGLisInitialized();

SAIGA_OPENGL_API int getVersionMajor();
SAIGA_OPENGL_API int getVersionMinor();
SAIGA_OPENGL_API void printOpenGLVersion();

SAIGA_OPENGL_API int getExtensionCount();
SAIGA_OPENGL_API bool hasExtension(const std::string& ext);
SAIGA_OPENGL_API std::vector<std::string> getExtensions();



enum class OpenGLVendor
{
    Nvidia,
    Ati,
    Intel,
    Mesa,
    Unknown
};

SAIGA_OPENGL_API OpenGLVendor getOpenGLVendor();

struct SAIGA_OPENGL_API OpenGLParameters
{
    enum class Profile
    {
        ANY,
        CORE,
        COMPATIBILITY
    };
    Profile profile = Profile::CORE;

    bool debug = true;

    // Throw an assertion if we get an opengl error.
    bool assertAtError = false;

    // all functionality deprecated in the requested version of OpenGL is removed
    bool forwardCompatible = false;

    int versionMajor = 3;
    int versionMinor = 2;

    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);
};

// called from OpenGLWindow::OpenGLWindow()
SAIGA_LOCAL void initSaigaGL(const OpenGLParameters& params);
SAIGA_LOCAL void cleanupSaigaGL();

}  // namespace Saiga

#define SAIGA_OPENGL_INCLUDED

#endif
