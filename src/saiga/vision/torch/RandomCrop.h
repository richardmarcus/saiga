/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/util/statistics.h"
#include "saiga/core/util/table.h"


namespace Saiga
{

inline IntrinsicsPinholef RandomImageCropX(ivec2 image_size_input, ivec2 image_size_crop, bool translate_to_border,
                                           bool random_translation, bool gaussian_sampling = false,
                                           vec2 min_max_zoom = vec2(1, 1))
{
    IntrinsicsPinholef K_crop = IntrinsicsPinholef();

    float delta_x = 0.0f;
    float zoom = 1.0f;

    {
        // Compute the minimum zoom level, considering only the x-axis.
        float min_zoom_x = static_cast<float>(image_size_crop.x()) / image_size_input.x();
        
        float cmin_zoom = std::max(min_zoom_x, min_max_zoom(0));
        float cmax_zoom = min_max_zoom(1);

        zoom = Random::sampleDouble(cmin_zoom, cmax_zoom);
    }

    // Max translation along x-axis (no translation along y-axis).
    float max_translation_x = static_cast<float>(image_size_input.x()) * zoom - image_size_crop.x();

    if (random_translation)
    {
        float min_sample_x = 0.0f;
        float max_sample_x = max_translation_x;

        if (translate_to_border)
        {
            float border_x = image_size_crop.x() * 0.5f;
            min_sample_x   = -border_x;
            max_sample_x   = max_translation_x + border_x;
        }

        if (!gaussian_sampling)
        {
            delta_x = Random::sampleDouble(min_sample_x, max_sample_x);
        }
        else
        {
            float len_2_x = 0.5f * (max_sample_x - min_sample_x);
            delta_x = Random::gaussRand(0, 0.5) * len_2_x + len_2_x + min_sample_x;
        }

        // Restrict delta to the valid range along x-axis.
        delta_x = std::max(0.0f, std::min(delta_x, max_translation_x));
    }
    else
    {
        delta_x = max_translation_x * 0.5f;
    }

    // Set zoom for x-axis and keep y-axis unchanged.
    K_crop.fx = zoom;
    K_crop.fy = zoom;  // We still zoom on both axes.

    // Set translation for x-axis only, fix y-axis translation to 0.
    K_crop.cx = -delta_x;
    K_crop.cy = 0.0f;

    return K_crop;
}

inline std::vector<IntrinsicsPinholef> RandomImageCropX(int N, int tries_per_crop, ivec2 image_size_input,
                                                        ivec2 image_size_crop, bool translate_to_border,
                                                        bool random_translation, bool gaussian_sampling = false,
                                                        vec2 min_max_zoom                  = vec2(1, 1),
                                                        int max_distance_from_image_center = -1)
{
    std::vector<float> centers_x;  // Store x-axis centers only.
    std::vector<IntrinsicsPinholef> res;
    
    for (int i = 0; i < N; ++i)
    {
        IntrinsicsPinholef best;
        float best_cx = 0.0f;
        float best_dis = -1;

        for (int j = 0; j < tries_per_crop; ++j)
        {
            auto intr = RandomImageCropX(image_size_input, image_size_crop, translate_to_border, random_translation,
                                         gaussian_sampling, min_max_zoom);

            // Compute center along x-axis only.
            float cx = image_size_crop.x() * 0.5f;
            cx       = intr.inverse().normalizedToImage(vec2(cx, 0)).x();  // Only use x component.

            float dis = 3573575737;
            for (const auto& cx2 : centers_x)
            {
                float d = (cx - cx2) * (cx - cx2);  // Squared distance along x-axis.
                if (d < dis)
                {
                    dis = d;
                }
            }

            if (centers_x.empty()) dis = 0;

            // Check if the x-center is within the allowed region.
            bool inside_sampling_region = max_distance_from_image_center < 0 ||
                                          std::abs(cx - image_size_input.x() * 0.5f) < max_distance_from_image_center;

            // Select the best crop.
            if (((j == 0 || dis > best_dis) && inside_sampling_region) || (j == (tries_per_crop - 1) && best_dis == -1))
            {
                best     = intr;
                best_cx  = cx;
                best_dis = dis;
            }
        }

        centers_x.push_back(best_cx);
        res.push_back(best);
    }
    return res;
}



// Computes the image crop as a homography matrix (returned as upper diagonal matrix).
inline IntrinsicsPinholef RandomImageCrop(ivec2 image_size_input, ivec2 image_size_crop, bool translate_to_border,
                                          bool random_translation, bool gaussian_sampling = false,
                                          vec2 min_max_zoom = vec2(1, 1))
{
    IntrinsicsPinholef K_crop = IntrinsicsPinholef();

    vec2 delta(0, 0);
    float zoom = 1.0f;

    {
        vec2 min_zoom_xy = image_size_crop.array().cast<float>() / image_size_input.array().cast<float>();

        float cmin_zoom = min_max_zoom(0);//std::max({min_zoom_xy(0), min_zoom_xy(1), min_max_zoom(0)});
        float cmax_zoom = min_max_zoom(1);

        zoom = Random::sampleDouble(cmin_zoom, cmax_zoom);
    }


    vec2 max_translation = image_size_input.cast<float>() * zoom - image_size_crop.cast<float>();

    if (random_translation)
    {
        vec2 min_sample = vec2(0, 0);
        vec2 max_sample = max_translation;

        if (translate_to_border)
        {
            vec2 border = image_size_crop.cast<float>() * 0.5f;
            min_sample  = -border;
            max_sample  = max_translation + border;
            // delta.x()   = Random::sampleDouble(-border.x(), max_translation.x() + border.x());
            // delta.y()   = Random::sampleDouble(-border.y(), max_translation.y() + border.y());
        }
        if (!gaussian_sampling)
        {
            delta.x() = Random::sampleDouble(min_sample.x(), max_sample.x());
            delta.y() = Random::sampleDouble(min_sample.y(), max_sample.y());
        }
        else
        {
            vec2 len_2 = 0.5 * (max_sample - min_sample);
            // 95% of samples are inside [-1,1]
            delta.x() = Random::gaussRand(0, 0.5) * len_2.x() + len_2.x() + min_sample.x();
            delta.y() = Random::gaussRand(0, 0.5) * len_2.y() + len_2.y() + min_sample.y();
        }
        delta = delta.array().max(vec2::Zero().array()).min(max_translation.array());
    }
    else
    {
        delta = max_translation * 0.5f;
    }

    K_crop.fx = zoom;
    K_crop.fy = zoom;

    K_crop.cx = -delta(0);
    K_crop.cy = -delta(1);

    return K_crop;
}

inline std::vector<IntrinsicsPinholef> RandomImageCrop(int N, int tries_per_crop, ivec2 image_size_input,
                                                       ivec2 image_size_crop, bool translate_to_border,
                                                       bool random_translation, bool gaussian_sampling = false,
                                                       vec2 min_max_zoom                  = vec2(1, 1),
                                                       int max_distance_from_image_center = -1)
{
    std::vector<vec2> centers;
    std::vector<IntrinsicsPinholef> res;
    for (int i = 0; i < N; ++i)
    {
        IntrinsicsPinholef best;
        vec2 best_c;
        float best_dis = -1;

        for (int j = 0; j < tries_per_crop; ++j)
        {
            auto intr = RandomImageCrop(image_size_input, image_size_crop, translate_to_border, random_translation,
                                        gaussian_sampling, min_max_zoom);

            vec2 c = image_size_crop.cast<float>() * 0.5f;
            c      = intr.inverse().normalizedToImage(c);

            float dis = 3573575737;
            for (auto& c2 : centers)
            {
                float d = (c - c2).squaredNorm();
                if (d < dis)
                {
                    dis = d;
                }
            }

            if (centers.empty()) dis = 0;

            // keep sample centers inside radius
            bool inside_sampling_region = max_distance_from_image_center < 0 ||
                                          length((c - vec2(image_size_input) / 2.f)) < max_distance_from_image_center;

            // only take crop if inside sampling region or if no valid sample was collected until the end
            if (((j == 0 || dis > best_dis) && inside_sampling_region) || (j == (tries_per_crop - 1) && best_dis == -1))
            {
                best     = intr;
                best_c   = c;
                best_dis = dis;
            }
        }

        centers.push_back(best_c);
        res.push_back(best);
    }
    return res;
}
// Computes the image crop as a homography matrix (returned as upper diagonal matrix).
inline IntrinsicsPinholef RandomCylinderCrop(ivec2 image_size_input, ivec2 image_size_crop, bool translate_to_border,
                                          bool random_translation, bool gaussian_sampling = false,
                                          vec2 min_max_zoom = vec2(1, 1))
{
    IntrinsicsPinholef K_crop = IntrinsicsPinholef();

    vec2 delta(0, 0);
    float zoom = 1.0f;

    {
        vec2 min_zoom_xy = image_size_crop.array().cast<float>() / image_size_input.array().cast<float>();

        float cmin_zoom = std::max({min_zoom_xy(0), min_zoom_xy(1), min_max_zoom(0)});
        float cmax_zoom = min_max_zoom(1);

        zoom = Random::sampleDouble(cmin_zoom, cmax_zoom);
    }


    vec2 max_translation = image_size_input.cast<float>() * zoom - image_size_crop.cast<float>();

    if (random_translation)
    {
        vec2 min_sample = vec2(0, 0);
        vec2 max_sample = max_translation;

        //cylinder image starts at the top left corner
        if (!translate_to_border)
        {
            //TODO_C actually do something different compared to img here?
            vec2 border = image_size_crop.cast<float>() * 0.5f;
            min_sample  = border;
            max_sample  = max_translation + 2*border;

        }
        if (!gaussian_sampling)
        {
            delta.x() = Random::sampleDouble(min_sample.x(), max_sample.x());
            delta.y() = Random::sampleDouble(min_sample.y(), max_sample.y());
        }
        else
        {
            vec2 len_2 = 0.5 * (max_sample - min_sample);
            // 95% of samples are inside [-1,1]
            delta.x() = Random::gaussRand(0, 0.5) * len_2.x() + len_2.x() + min_sample.x();
            delta.y() = Random::gaussRand(0, 0.5) * len_2.y() + len_2.y() + min_sample.y();
        }
        delta = delta.array().max(vec2::Zero().array()).min(max_translation.array());
    }
    else
    {
        delta = max_translation * 0.5f;
    }

    K_crop.fx = zoom;
    K_crop.fy = zoom;

    K_crop.cx = delta(0);
    K_crop.cy = delta(1);
    K_crop.s = 0.0 ;

    return K_crop;
}

//not used because it is the same as RandomImageCrop TODO_C
inline std::vector<IntrinsicsPinholef> RandomCylinderCrop(int N, int tries_per_crop, ivec2 image_size_input,
                                                       ivec2 image_size_crop, bool translate_to_border,
                                                       bool random_translation, bool gaussian_sampling = false,
                                                       vec2 min_max_zoom                  = vec2(1, 1),
                                                       int max_distance_from_image_center = -1)
{
    std::vector<vec2> centers;
    std::vector<IntrinsicsPinholef> res;
    for (int i = 0; i < N; ++i)
    {
        IntrinsicsPinholef best;
        vec2 best_c;
        float best_dis = -1;

        for (int j = 0; j < tries_per_crop; ++j)
        {
            auto intr = RandomCylinderCrop(image_size_input, image_size_crop, translate_to_border, random_translation,
                                        gaussian_sampling, min_max_zoom);

            vec2 c = vec2(intr.cx, intr.cy) ;

            float dis = 3573575737;
            for (auto& c2 : centers)
            {
                float d = (c - c2).squaredNorm();
                if (d < dis)
                {
                    dis = d;
                }
            }
            if (centers.empty()) dis = 0;

            // keep sample centers inside radius
            bool inside_sampling_region = max_distance_from_image_center < 0 ||
                                          length(c) < max_distance_from_image_center;

            // only take crop if inside sampling region or if no valid sample was collected until the end
            if (((j == 0 || dis > best_dis) && inside_sampling_region) || (j == (tries_per_crop - 1) && best_dis == -1))
            {
                best     = intr;
                best_c   = c;
                best_dis = dis;
            }
        }

        centers.push_back(best_c);
        res.push_back(best);
    }
    return res;
}



}  // namespace Saiga