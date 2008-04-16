/*! \file CudaShadingInterface.h
 *  \author Jared Hoberock
 *  \brief Defines an interface that Cuda shaders
 *         interact with to perform various common shading
 *         functions.
 */

#pragma once

#include "../shading/CudaScatteringDistributionFunction.h"

#include "../shading/CudaNullScattering.h"
#include "../shading/CudaLambertian.h"
#include "../shading/CudaHemisphericalEmission.h"
#include "../shading/CudaPerspectiveSensor.h"

class CudaShadingInterface
{
  public:
    static inline __host__ __device__ void null(CudaScatteringDistributionFunction &f)
    {
      f.mType = NULL_SCATTERING;
    } // end null()

    static inline __host__ __device__ CudaScatteringDistributionFunction null(void)
    {
      CudaScatteringDistributionFunction result;
      CudaShadingInterface::null(result);
      return result;
    } // end null()

    static inline __host__ __device__ void diffuse(const float3 &Kd, CudaScatteringDistributionFunction &f)
    {
      CudaLambertian *cl = reinterpret_cast<CudaLambertian*>(&f.mFunction);
      *cl = CudaLambertian(Kd);
      f.mType = LAMBERTIAN;
    } // end diffuse()

    static inline __host__ __device__ CudaScatteringDistributionFunction diffuse(const float3 &Kd)
    {
      CudaScatteringDistributionFunction result;
      CudaShadingInterface::diffuse(Kd, result);
      return result;
    } // end diffuse()

    static inline __host__ __device__ void hemisphericalEmission(const float3 &radiosity, CudaScatteringDistributionFunction &f)
    {
      CudaHemisphericalEmission *che = reinterpret_cast<CudaHemisphericalEmission*>(&f.mFunction);
      *che = CudaHemisphericalEmission(radiosity);
      f.mType = HEMISPHERICAL_EMISSION;
    } // end hemisphericalEmission()

    static inline __host__ __device__ CudaScatteringDistributionFunction hemisphericalEmission(const float3 &radiosity)
    {
      CudaScatteringDistributionFunction result;
      CudaShadingInterface::hemisphericalEmission(radiosity, result);
      return result;
    } // end hemisphericalEmission()

    static inline __host__ __device__ void perspectiveSensor(const float3 &Ks, const float aspect, const float3 &origin, CudaScatteringDistributionFunction &f)
    {
      CudaPerspectiveSensor *cps = reinterpret_cast<CudaPerspectiveSensor*>(&f.mFunction);
      *cps = CudaPerspectiveSensor(Ks,aspect,origin);
      f.mType = PERSPECTIVE_SENSOR;
    } // end perspectiveSensor()

    static inline __host__ __device__ CudaScatteringDistributionFunction perspectiveSensor(const float3 &Ks, const float aspect, const float3 &origin)
    {
      CudaScatteringDistributionFunction result;
      CudaShadingInterface::perspectiveSensor(Ks,aspect,origin,result);
      return result;
    } // end perspectiveSensor()
}; // end CudaShadingInterface

