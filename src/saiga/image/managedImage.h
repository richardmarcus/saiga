﻿/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/image/imageBase.h"
#include "saiga/image/imageView.h"
#include "saiga/image/imageFormat.h"
#include <vector>

namespace Saiga {

#define DEFAULT_ALIGNMENT 4

class SAIGA_GLOBAL Image : public ImageBase
{
public:
    using byte_t = unsigned char;


    ImageType type = TYPE_UNKNOWN;
protected:
    std::vector<byte_t> vdata;

public:

    Image(){}
    Image(ImageType type) : type(type) {}
    Image(int h, int w , ImageType type);
    Image(std::string file) { load(file); }

    // Note: This creates a copy of img
    template<typename T>
    Image(ImageView<T> img)
    {
        setFormatFromImageView(img);
        create();
        img.copyTo(getImageView<T>());
    }

    void create();
    void create(int h, int w);
    void create(int h, int w, ImageType t);

    void free();
    /**
     * @brief makeZero
     * Sets all data to 0.
     */
    void makeZero();

    /**
     * @brief valid
     * Checks if this image has at least 1 pixel and a valid type.
     */
    bool valid();

    void* data() { return vdata.data(); }
    uint8_t* data8() { return vdata.data(); }

    template<typename T>
    inline
    T& at(int y, int x)
    {
        return reinterpret_cast<T*>(rowPtr(y))[x];
    }

    inline
    void* rowPtr(int y)
    {
        auto ptr = data8() + y * pitchBytes;
        return ptr;
    }


    template<typename T>
    ImageView<T> getImageView()
    {
        SAIGA_ASSERT(elementSize(type) == sizeof(T));
        ImageView<T> res(*this);
        res.data = data();
        return res;
    }

    template<typename T>
    void setFormatFromImageView(ImageView<T> v)
    {
        ImageBase::operator=(v);
        type = ImageTypeTemplate<T>::type;
    }

    bool load(const std::string &path);
    bool save(const std::string &path);

    SAIGA_GLOBAL friend std::ostream& operator<<(std::ostream& os, const Image& f);
};


/**
 * Converts a floating point image to a 8-bit image and saves it.
 * Useful for debugging.
 */
SAIGA_GLOBAL bool saveHSV(const std::string& path, ImageView<float> img, float vmin, float vmax);
SAIGA_GLOBAL bool save(const std::string& path, ImageView<float> img, float vmin, float vmax);



}
