/*! \file CudaIntersection.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an Intersection
 *         class which is compatible with CUDA.
 */

#pragma once

#include <stdcuda/stdcuda.h>

// XXX hack total
#define inline inline __host__ __device__
#include "Intersection.h"
#undef inline

typedef IntersectionBase<float3,float3,float2,float3> CudaIntersection;

