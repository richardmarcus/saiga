/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "TorchHelper.h"

#include "cuda_runtime.h"


#ifdef __CUDACC__

#    ifndef TINY_TORCH
// Converts the torch half type to the CUDA build-in half type
// Only available from cuda files
template <>
inline __half* at::TensorBase::data_ptr<__half>() const
{
    return (__half*)data_ptr<torch::Half>();
}
#    endif
#endif

namespace Saiga
{
template <typename T, int dim, typename IndexType = int64_t, bool CUDA = true>
struct StaticDeviceTensor
{
    T* __restrict__ data = nullptr;

    IndexType sizes[dim];
    IndexType strides[dim];

    StaticDeviceTensor() = default;

    StaticDeviceTensor(torch::Tensor t)
    {
        if (!t.defined())
        {
            data = nullptr;
            for (int i = 0; i < dim; ++i)
            {
                sizes[i]   = 0;
                strides[i] = 0;
            }
            return;
        }
        if (CUDA)
        {
            SAIGA_ASSERT(t.is_cuda());
        }
        else
        {
            SAIGA_ASSERT(t.is_cpu());
        }
        SAIGA_ASSERT(t.dim() == dim);
        data = t.template data_ptr<T>();
        for (int i = 0; i < dim; ++i)
        {
            sizes[i]   = t.size(i);
            strides[i] = t.stride(i);
        }
    }

    template <typename G>
    HD StaticDeviceTensor<G, dim, IndexType, CUDA> cast()
    {
        StaticDeviceTensor<G, dim, IndexType, CUDA> result;
        result.data = reinterpret_cast<G*>(data);

        for (int i = 0; i < dim - 1; ++i)
        {
            result.sizes[i] = sizes[i];

            if constexpr (sizeof(G) > sizeof(T))
            {
                // CHECK_EQ(strides[i] % (sizeof(G) / sizeof(T)), 0);
                result.strides[i] = strides[i] / (sizeof(G) / sizeof(T));
            }
            else
            {
                result.strides[i] = strides[i] * (sizeof(T) / sizeof(G));
            }
        }

        if constexpr (sizeof(G) > sizeof(T))
        {
            // CHECK_EQ(strides[dim - 1], 1);
            // CHECK_EQ(result.sizes[dim - 1] % (sizeof(G) / sizeof(T)), 0);
            result.sizes[dim - 1]   = sizes[dim - 1] / (sizeof(G) / sizeof(T));
            result.strides[dim - 1] = 1;
        }
        else
        {
            result.sizes[dim - 1]   = sizes[dim - 1] * (sizeof(T) / sizeof(G));
            result.strides[dim - 1] = 1;
        }

        return result;
    }

    HD inline IndexType size(IndexType i)
    {
        CUDA_KERNEL_ASSERT(i < dim);
        return sizes[i];
    }

    HD inline int64_t numel()
    {
        int64_t count = 1;
        for (int i = 0; i < dim; ++i)
        {
            count *= sizes[i];
        }
        return count;
    }

    HD inline IndexType Offset(std::array<IndexType, dim> indices)
    {
        CUDA_KERNEL_ASSERT(data);
        // The array offset is always 64 bit and does not depend on the index type
        IndexType index = 0;
        for (int i = 0; i < dim; ++i)
        {
            CUDA_KERNEL_ASSERT(indices[i] >= 0 && indices[i] < sizes[i]);
            index += strides[i] * indices[i];
        }
        return index;
    }

    HD inline T& Get(std::array<IndexType, dim> indices)
    {
        IndexType index = 0;
        for (int i = 0; i < dim; ++i)
        {
#if defined(CUDA_DEBUG) && defined(__CUDACC__)
            CUDA_KERNEL_ASSERT(indices[i] >= 0);
            CUDA_KERNEL_ASSERT(indices[i] < sizes[i]);
#endif
            index += strides[i] * indices[i];
        }
        return data[index];
    }

    HD inline T* GetPtr(std::array<IndexType, dim> indices)
    {
        int64_t index = 0;
        for (int i = 0; i < dim; ++i)
        {
            index += (int64_t)strides[i] * (int64_t)indices[i];
        }
        return &data[index];
    }

    template <typename... Ts>
    HD inline T& operator()(Ts... args)
    {
        return Get({args...});
    }

    HD inline ImageDimensions Image()
    {
        static_assert(dim >= 2, "must have atleast 2 dimensions to be an image");
        return ImageDimensions(sizes[dim - 2], sizes[dim - 1]);
    }

    HD inline bool defined() { return data != nullptr; }

    void Print()
    {
        std::cout << "StaticDeviceTensor<" << typeid(T).name() << "," << dim << "," << typeid(IndexType).name() << ","
                  << CUDA << "> [";
        for (int i = 0; i < dim; ++i)
        {
            std::cout << sizes[i] << ",";
        }
        std::cout << "] [";
        for (int i = 0; i < dim; ++i)
        {
            std::cout << strides[i] << ",";
        }
        std::cout << "]\n";
    }
};

template <typename T, int dim, typename IndexType = int64_t, bool CUDA = true>
struct ConstReadOnlyStaticDeviceTensor
{
    const T* __restrict__ data;
    IndexType sizes[dim];
    IndexType strides[dim];

    // cuda does not allow const __restrict__ pointers in structs
    // see:
    // https://forums.developer.nvidia.com/t/restrict-seems-to-be-ignored-for-base-pointers-in-structs-having-base-pointers-with-restrict-as-kernel-arguments-directly-works-as-expected/154020/12?u=lkskstlr
    // thus the data pointer has to be passed directly to the kernel and can then be set into this structure
    __device__ void setDataPointer(const T* __restrict__ d_ptr) { data = d_ptr; }

    ConstReadOnlyStaticDeviceTensor() = default;

    ConstReadOnlyStaticDeviceTensor(torch::Tensor t)
    {
        if (!t.defined() || t.numel() == 0)
        {
            // data = nullptr;
            for (int i = 0; i < dim; ++i)
            {
                sizes[i]   = 0;
                strides[i] = 0;
            }
            return;
        }
        if (CUDA)
        {
            SAIGA_ASSERT(t.is_cuda());
        }
        else
        {
            SAIGA_ASSERT(t.is_cpu());
        }
        SAIGA_ASSERT(t.dim() == dim);
        // data = t.template data_ptr<T>();
        for (int i = 0; i < dim; ++i)
        {
            sizes[i]   = t.size(i);
            strides[i] = t.stride(i);
        }
    }

    HD inline IndexType size(IndexType i)
    {
        CUDA_KERNEL_ASSERT(i < dim);
        return sizes[i];
    }

    // same as get but with bounds checks
    HD inline const T& At(std::array<IndexType, dim> indices)
    {
        CUDA_KERNEL_ASSERT(data);
        // The array offset is always 64 bit and does not depend on the index type
        int64_t index = 0;
        for (int i = 0; i < dim; ++i)
        {
            CUDA_KERNEL_ASSERT(indices[i] >= 0 && indices[i] < sizes[i]);
            index += (int64_t)strides[i] * (int64_t)indices[i];
        }
        return data[index];
    }

    HD inline const T& Get(std::array<IndexType, dim> indices)
    {
        int64_t index = 0;
        for (int i = 0; i < dim; ++i)
        {
            index += (int64_t)strides[i] * (int64_t)indices[i];
        }
        return data[index];
    }

    template <typename... Ts>
    HD inline const T& operator()(Ts... args)
    {
        return Get({args...});
    }

    HD inline ImageDimensions Image()
    {
        static_assert(dim >= 2, "must have atleast 2 dimensions to be an image");
        return ImageDimensions(sizes[dim - 2], sizes[dim - 1]);
    }
};
}  // namespace Saiga
