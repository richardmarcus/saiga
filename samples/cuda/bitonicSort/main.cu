﻿/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <iostream>
#include <vector>
#include "saiga/util/glm.h"
#include "saiga/util/table.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/bitonicSort.h"
#include "saiga/cuda/pinned_vector.h"


using Saiga::ArrayView;
using Saiga::CUDA::ThreadInfo;

using Saiga::CUDA::bfe;
using Saiga::CUDA::shuffleSwapCompare;


template<typename T>
inline __device__
T bitonicSortStageSimple(T v, unsigned int stage,  unsigned int l)
{
    for(int i = stage; i >= 0; --i)
    {
        auto distance = 1 << i;
        unsigned int direction;

        direction = bfe(l,i) ^ bfe(l,stage+1);
        v = shuffleSwapCompare(v,distance,direction);
    }
    return v;
}

template<typename T>
inline __device__
T bitonicWarpSortSimple(T v, unsigned int l)
{
    for(int stage = 0; stage < 5 ; ++stage)
    {
        v = bitonicSortStageSimple(v,stage,l);
    }
    return v;
}

template<typename T>
__global__ static
void bitonicSortSimple(ArrayView<T> data)
{
    ThreadInfo<> ti;
    if(ti.thread_id >= data.size()) return;

    auto l = ti.lane_id;
    auto v = data[ti.thread_id];
    v = bitonicWarpSortSimple<T>(v,l);
    data[ti.thread_id] = v;
}



template<typename T, unsigned int SIZE = 32>
__global__ static
void bitonicSortSaiga(ArrayView<T> data)
{
    ThreadInfo<> ti;
    if(ti.thread_id >= data.size()) return;

    auto l = ti.lane_id;
    auto v = data[ti.thread_id];
    v = Saiga::CUDA::bitonicWarpSort<T,SIZE>(v,l);
    data[ti.thread_id] = v;
}



static void bitonicSortTest()
{
    int N = 64;
    using T = float;
    Saiga::thrust::pinned_vector<T> h_data(N), res;
    thrust::device_vector<T> d_data(N);

    for(auto& f : h_data)
    {
        f = rand() % N;
    }


    Saiga::Table table({6,7,7});

    {
        cout << "Full warp sort (width = 32)" << endl;
        table << "Id" << "Before" << "After";
        d_data = h_data;
        bitonicSortSimple<T><<<1,N>>>(d_data);
        res = d_data;
        for(int i = 0; i < N; ++i)
        {
            table << i << h_data[i] << res[i];
        }
    }

    {
        cout << "Partial warp sort (width = 8)" << endl;
        table << "Id" << "Before" << "After";
        d_data = h_data;
        bitonicSortSaiga<T,8><<<1,N>>>(d_data);
        res = d_data;
        for(int i = 0; i < N; ++i)
        {
            table << i << h_data[i] << res[i];
        }
    }

}

int main(int argc, char *argv[])
{
    bitonicSortTest();

    cout << "Done." << endl;
}

