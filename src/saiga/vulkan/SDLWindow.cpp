﻿/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "SDLWindow.h"
#include "SDL2/SDL.h"
#include "SDL2/SDL_vulkan.h"

namespace Saiga {
namespace Vulkan {

std::vector<const char *> SDLWindow::getRequiredInstanceExtensions()
{
    unsigned int count = 0;
    const char **names = NULL;
    auto res = SDL_Vulkan_GetInstanceExtensions(sdl_window, &count, NULL);
    cout << SDL_GetError() << endl;
    SAIGA_ASSERT(res);
    // now count is (probably) 2. Now you can make space:
    names = new const char *[count];

    // now call again with that not-NULL array you just allocated.
    res = SDL_Vulkan_GetInstanceExtensions(sdl_window, &count, names);
    cout << SDL_GetError() << endl;
SAIGA_ASSERT(res);
    cout << "num extensions " << count << endl;
    // Now names should have (count) strings in it:

    std::vector<const char *> extensions;
    for (unsigned int i = 0; i < count; i++) {
        printf("Extension %d: %s\n", i, names[i]);
        extensions.push_back(names[i]);
    }

    // use it for VkInstanceCreateInfo and when you're done, free it:

    delete[] names;

    return extensions;
}

void SDLWindow::setupWindow()
{

    {
        //Initialize SDL
        if( SDL_Init( SDL_INIT_VIDEO ) < 0 ){
            std::cout << "SDL could not initialize! SDL Error: " << SDL_GetError() << std::endl;

        }

        sdl_window = SDL_CreateWindow("asdf", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_VULKAN );
//        std::cout << "SDL could not initialize! SDL Error: " << SDL_GetError() << std::endl;
        SAIGA_ASSERT(sdl_window);
        // (you should check return values for errors in all this, but whatever.)
    }


}

void SDLWindow::createSurface(VkInstance instance, VkSurfaceKHR *surface)
{
    auto asdf = SDL_Vulkan_CreateSurface(sdl_window,instance,surface);
    SAIGA_ASSERT(asdf);
}

}
}
