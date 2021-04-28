/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "tone_mapper.h"

#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/shader/shaderLoader.h"


namespace Saiga
{
ToneMapper::ToneMapper()
{
    shader = shaderLoader.load<Shader>("tone_map.glsl");
    uniforms.create(ArrayView<TonemapParameters>(params), GL_STATIC_DRAW);

    camera_response.MakeGamma(1.0 / 2.2);
    camera_response.normalize(1);
    params_dirty = true;
}
void ToneMapper::Map(Texture* input_hdr_color_image, Texture* output_ldr_color_image)
{
    if (params_dirty)
    {
        uniforms.update(ArrayView<TonemapParameters>(params));
        response_texture = std::make_shared<Texture1D>();
        response_texture->create(camera_response.irradiance.size(), GL_RED, GL_R32F, GL_FLOAT,
                                 camera_response.irradiance.data());
        params_dirty = false;
    }

    shader->bind();
    input_hdr_color_image->bindImageTexture(0, GL_READ_ONLY);
    output_ldr_color_image->bindImageTexture(1, GL_WRITE_ONLY);
    // response_texture->bindImageTexture(2, GL_READ_ONLY);
    shader->upload(2, response_texture.get(), 0);
    uniforms.bind(3);
    int gw = iDivUp(input_hdr_color_image->getWidth(), 16);
    int gh = iDivUp(input_hdr_color_image->getHeight(), 16);
    shader->dispatchCompute(uvec3(gw, gh, 1));
    shader->unbind();
}

vec3 ColorTemperatureToRGB(float temperatureInKelvins)
{
    vec3 retColor;

    temperatureInKelvins = clamp(temperatureInKelvins, 1000.0, 40000.0) / 100.0;

    if (temperatureInKelvins <= 66.0)
    {
        retColor(0) = 1.0;
        retColor(1) = saturate(0.39008157876901960784 * log(temperatureInKelvins) - 0.63184144378862745098);
    }
    else
    {
        float t     = temperatureInKelvins - 60.0;
        retColor(0) = saturate(1.29293618606274509804 * pow(t, -0.1332047592));
        retColor(1) = saturate(1.12989086089529411765 * pow(t, -0.0755148492));
    }

    if (temperatureInKelvins >= 66.0)
        retColor(2) = 1.0;
    else if (temperatureInKelvins <= 19.0)
        retColor(2) = 0.0;
    else
        retColor(2) = saturate(0.54320678911019607843 * log(temperatureInKelvins - 10.0) - 1.19625408914);

    return retColor;
}


void ToneMapper::imgui()
{
    params_dirty |= ImGui::SliderFloat("exposure", &params.exposure, 0.1, 5);
    params_dirty |= ImGui::SliderFloat3("vignette_coeffs", params.vignette_coeffs.data(), -3, 1);
    params_dirty |= ImGui::SliderFloat2("vignette_offset", params.vignette_offset.data(), -1, 1);

    ImGui::Separator();
    ImGui::Text("Camera Response");

    if (ImGui::Button("gamma = 2.2"))
    {
        camera_response.MakeGamma(2.2);
        params_dirty = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("gamma = 1"))
    {
        camera_response.MakeGamma(1);
        params_dirty = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("gamma = 1.0/2.2"))
    {
        camera_response.MakeGamma(1.0 / 2.2);
        params_dirty = true;
    }
    static float gamma = 1.0 / 1.5;
    ImGui::SliderFloat("gamma", &gamma, 0, 4);
    if (ImGui::Button("gamma response"))
    {
        camera_response.MakeGamma(gamma);
        params_dirty = true;
    }

    ImGui::PlotLines("###response", camera_response.irradiance.data(), camera_response.irradiance.size(), 0, "", 0, 1,
                     ImVec2(100, 80));

    ImGui::Separator();
    ImGui::Text("White Balance");
    if (ImGui::ColorEdit3("white_point", params.white_point.data()))
    {
        params_dirty = true;
    }
    if (ImGui::SliderFloat("color_temperature", &color_temperature, 1000, 15000))
    {
        params.white_point = ColorTemperatureToRGB(color_temperature);
        //        params.white_point = vec3(1, 1, 1).array() / params.white_point.array();
        //        float max_l        = params.white_point.maxCoeff();
        //        params.white_point /= max_l;
        params_dirty = true;
    }
}
}  // namespace Saiga
