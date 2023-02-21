/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "TorchHelper.h"

#include <torch/script.h>

#ifdef SAIGA_USE_TORCHVISION

#    if __has_include(<torchvision/csrc/models/vgg.h>)
// Use this if torchvision was added as a submodule
#        include <torchvision/csrc/models/vgg.h>
#    else
// System path otherwise
// #    include <torchvision/models/vgg.h>

#    endif
#    include <torchvision/csrc/vision.h>

namespace Saiga
{
// This is a helper class to get the pytorch pretrained vgg loss into c++.
// First, run the following code in python to extract the vgg weights:
//      model =  torchvision.models.vgg19(pretrained=True).features
//      torch.jit.save(torch.jit.script(model), 'vgg_script.pth')
// After that you can create the vgg loss object in c++ and load the weights with:
//      PretrainedVGG19Loss loss("vgg_script.pth");
//
// After that, this loss can be used in regular python code :)
class PretrainedVGG19LossImpl : public torch::nn::Module
{
   public:
    PretrainedVGG19LossImpl(const std::string& file, bool use_average_pool = true, bool from_pytorch = true)
    {
        torch::nn::Sequential features = vision::models::VGG19()->features;

        if (use_average_pool)
        {
            for (auto m : *features)
            {
                auto mod = m.ptr();

                if (auto func_ptr = std::dynamic_pointer_cast<torch::nn::FunctionalImpl>(mod))
                {
                    auto x = torch::zeros({1, 1, 4, 4});
                    x      = func_ptr->forward(x);
                    if (x.size(2) == 2)
                    {
                        seq->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2})));
                    }
                    else
                    {
                        seq->push_back(m);
                    }
                }
                else
                {
                    seq->push_back(m);
                }
            }
        }
        else
        {
            seq = features;
        }

        if (from_pytorch)
        {
            {
                float array[] = {0.485, 0.456, 0.406};
                auto options  = torch::TensorOptions().dtype(torch::kFloat32);
                mean_         = torch::from_blob(array, {1, 3, 1, 1}, options).clone();
            }

            {
                float array[] = {0.229, 0.224, 0.225};
                auto options  = torch::TensorOptions().dtype(torch::kFloat32);
                std_          = torch::from_blob(array, {1, 3, 1, 1}, options).clone();
            }
        }
        else
        {
            {
                float array[] = {103.939 / 255.f, 116.779 / 255.f, 123.680 / 255.f};
                auto options  = torch::TensorOptions().dtype(torch::kFloat32);
                mean_         = torch::from_blob(array, {1, 3, 1, 1}, options).clone();
            }

            {
                float array[] = {1. / 255, 1. / 255, 1. / 255};
                auto options  = torch::TensorOptions().dtype(torch::kFloat32);
                std_          = torch::from_blob(array, {1, 3, 1, 1}, options).clone();
            }
        }

        std::cout << "num layers in vgg " << seq->size() << std::endl;

        if (0)
        {
            layers     = {1, 3, 6, 8, 11, 13, 15};
            last_layer = 15;
        }
        else
        {
            layers     = {1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29};
            last_layer = 29;
        }

        torch::nn::Sequential cpy;

        int i = 0;
        for (auto l : *seq)
        {
            cpy->push_back(l);
            if (i >= last_layer) break;
            i++;
        }
        seq = cpy;
        seq->eval();
        SAIGA_ASSERT(seq->size() == last_layer + 1);


        LoadFromPythonExport(file);

        register_module("features", seq);
        register_buffer("mean", mean_);
        register_buffer("std", std_);
    }

    void LoadFromPythonExport(std::string file)
    {
        torch::jit::Module py_model = torch::jit::load(file);
        auto params                 = py_model.named_parameters();
        std::map<std::string, torch::Tensor> param_map;
        for (auto p : params)
        {
            param_map[p.name] = p.value;
        }

        torch::nn::Module module = *seq;
        for (int i = 0; i < module.children().size(); ++i)
        {
            if (auto conv_ptr = module.children()[i]->as<torch::nn::Conv2d>())
            {
                auto weight = param_map[std::to_string(i) + ".weight"];
                auto bias   = param_map[std::to_string(i) + ".bias"];
                SAIGA_ASSERT(conv_ptr->weight.sizes() == weight.sizes());
                SAIGA_ASSERT(conv_ptr->bias.sizes() == bias.sizes());
                {
                    torch::NoGradGuard ng;
                    conv_ptr->weight.set_(weight);
                    conv_ptr->bias.set_(bias);
                }
            }
        }
    }

    torch::Tensor normalize_inputs(torch::Tensor x) { return (x - mean_) / std_; }

    torch::Tensor forward(torch::Tensor input, torch::Tensor target)
    {
        torch::Tensor features_input  = normalize_inputs(input);
        torch::Tensor features_target = normalize_inputs(target);


        torch::Tensor loss = torch::zeros({1}, torch::TensorOptions().device(input.device()));

        int i = 0;
        for (auto m : *seq)
        {
            features_input  = m.any_forward(features_input).get<torch::Tensor>();
            features_target = m.any_forward(features_target).get<torch::Tensor>();

            if (0)
            {
                auto mod = m.get<torch::nn::Conv2d>();

                float f = features_input[0][0][0][0].item().toFloat();

                std::cout << "f " << f << std::endl;

                PrintTensorInfo(features_input);
                PrintTensorInfo(features_target);
                PrintTensorInfo(mod->weight);
                PrintTensorInfo(mod->bias);

                //                std::cout << mod->weight.mean().item().to<float>() << " " <<
                //                mod->bias.mean().item().to<float>()
                //                          << std::endl;
                //                std::cout << "input,target mean: " << features_input.mean().item().to<float>() << " "
                //                          << features_target.mean().item().to<float>() << std::endl;
                exit(0);
            }

            if (layers.count(i) > 0)
            {
                loss = loss + torch::nn::functional::l1_loss(features_input, features_target);
            }

            if (i >= last_layer)
            {
                break;
            }
            i++;
        }

        return loss;
    }

    int last_layer;
    torch::nn::Sequential seq;

    std::set<int> layers;
    torch::Tensor mean_, std_;
};

TORCH_MODULE(PretrainedVGG19Loss);

#else
namespace Saiga
{
// ########### Python VGG class
//      import torch
//      import torch.nn.functional as F
//      import torch.nn as nn
//      import torchvision
//      from collections import OrderedDict
//      import os
//      from os.path import join
// class VGGLoss(nn.Module):
//    def __init__(self, net='caffe', partialconv=False, optimized=False, save_dir='.cache/torch/models'):
//        super().__init__()
//
//        self.partialconv = partialconv
//
//        if net == 'pytorch':
//            vgg19 = torchvision.models.vgg19(pretrained=True).features
//
//            self.register_buffer('mean_', torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None])
//            self.register_buffer('std_', torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None])
//
//        elif net == 'caffe':
//            if not os.path.exists(join(save_dir, 'vgg_caffe_features.pth')):
//                vgg_weights =
//                torch.utils.model_zoo.load_url('https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth',
//                model_dir=save_dir)
//
//                map = {'classifier.6.weight':u'classifier.7.weight', 'classifier.6.bias':u'classifier.7.bias'}
//                vgg_weights = OrderedDict([(map[k] if k in map else k,v) for k,v in vgg_weights.items()])
//
//                model = torchvision.models.vgg19()
//                model.classifier = nn.Sequential(View(), *model.classifier._modules.values())
//
//                model.load_state_dict(vgg_weights)
//
//                vgg19 = model.features
//                os.makedirs(save_dir, exist_ok=True)
//                torch.save(vgg19, join(save_dir, 'vgg_caffe_features.pth'))
//
//                self.register_buffer('mean_', torch.FloatTensor([103.939, 116.779, 123.680])[None, :, None, None] /
//                255.) self.register_buffer('std_', torch.FloatTensor([1./255, 1./255, 1./255])[None, :, None, None])
//
//            else:
//                vgg19 = torch.load(join(save_dir, 'vgg_caffe_features.pth'))
//                self.register_buffer('mean_', torch.FloatTensor([103.939, 116.779, 123.680])[None, :, None, None] /
//                255.) self.register_buffer('std_', torch.FloatTensor([1./255, 1./255, 1./255])[None, :, None, None])
//        else:
//            assert False
//
//        if self.partialconv:
//            part_conv = PartialConv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
//            part_conv.weight = vgg19[0].weight
//            part_conv.bias = vgg19[0].bias
//            vgg19[0] = part_conv
//
//        vgg19_avg_pooling = []
//
//
//        for weights in vgg19.parameters():
//            weights.requires_grad = False
//
//        for module in vgg19.modules():
//            if module.__class__.__name__ == 'Sequential':
//                continue
//            elif module.__class__.__name__ == 'MaxPool2d':
//                vgg19_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
//            else:
//                vgg19_avg_pooling.append(module)
//
//        if optimized:
//            self.layers = [3, 8, 17, 26, 35]
//        else:
//            self.layers = [1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29]
//
//        self.vgg19 = nn.Sequential(*vgg19_avg_pooling)
//
//    def normalize_inputs(self, x):
//        return (x - self.mean_) / self.std_
//
//    def forward(self, input, target):
//        loss = 0
//
//        if self.partialconv:
//            eps = 1e-9
//            mask = target.sum(1, True) > eps
//            mask = mask.float()
//
//        features_input = self.normalize_inputs(input)
//        features_target = self.normalize_inputs(target)
//        for i, layer in enumerate(self.vgg19):
//            features_input  = layer(features_input)
//            features_target = layer(features_target)
//
//            if i in self.layers:
//                loss = loss + F.l1_loss(features_input, features_target)
//
//        return loss
//
// ############## Python Tracing:
//
//      model = VGGLoss()
//      traced_script_module = torch.jit.trace(model, (example, example))
//      traced_script_module.save("traced_caffe_vgg.pt")
// ################## or::
//      model2 = VGGLoss('pytorch')
//      traced_script_module = torch.jit.trace(model2, (example, example))
//      traced_script_module.save("traced_vgg.pt")
//
// ################ C++ Usage:
//
//     PretrainedVGG19Loss vgg("traced_vgg.pt");
//     auto loss = lpips.forward(output, target);
//
class PretrainedVGG19Loss
{
   public:
    PretrainedVGG19Loss(const std::string& file) { module = torch::jit::load(file); }

    torch::Tensor forward(torch::Tensor input, torch::Tensor target)
    {
        SAIGA_ASSERT(input.dim() == 4);

        module.to(input.device());
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        inputs.push_back(target);
        return module.forward(inputs).toTensor();
    }
    void to(torch::Device d) { module.to(d); }
    void eval() { module.eval(); }
    torch::jit::script::Module module;
};
// TORCH_MODULE(PretrainedVGG19Loss);

#endif
}  // namespace Saiga
