/*! \file stdcudamaterial.h
 *  \author Jared Hoberock
 *  \brief Thunk functions for hiding ShadingInterface nastiness from shaders.
 */

#include "CudaShadingInterface.h"

inline __host__ __device__ CudaScatteringDistributionFunction null(void)
{
  return CudaShadingInterface::null();
} // end null()

inline __host__ __device__ void null(CudaScatteringDistributionFunction &f)
{
  return CudaShadingInterface::null(f);
} // end null()

inline __host__ __device__ CudaScatteringDistributionFunction diffuse(const float3 &Kd)
{
  return CudaShadingInterface::diffuse(Kd);
} // end diffuse()

inline __host__ __device__ void diffuse(const float3 &Kd, CudaScatteringDistributionFunction &f)
{
  return CudaShadingInterface::diffuse(Kd, f);
} // end diffuse()

inline __host__ __device__ CudaScatteringDistributionFunction hemisphericalEmission(const float3 &radiosity)
{
  return CudaShadingInterface::hemisphericalEmission(radiosity);
} // end hemisphericalEmission()

inline __host__ __device__ void hemisphericalEmission(const float3 &radiosity, CudaScatteringDistributionFunction &f)
{
  return CudaShadingInterface::hemisphericalEmission(radiosity, f);
} // end hemisphericalEmission()

inline __host__ __device__ CudaScatteringDistributionFunction perspectiveSensor(const float3 &Ks, const float aspect, const float3 &lowerLeft)
{
  return CudaShadingInterface::perspectiveSensor(Ks,aspect,lowerLeft);
} // end perspectiveSensor()

inline __host__ __device__ void perspectiveSensor(const float3 &Ks, const float aspect, const float3 &lowerLeft, CudaScatteringDistributionFunction &f)
{
  return CudaShadingInterface::perspectiveSensor(Ks,aspect,lowerLeft,f);
} // end perspectiveSensor()

inline __host__ __device__ CudaScatteringDistributionFunction refraction(const float3 &Kt, const float etai, const float etat)
{
  return CudaShadingInterface::refraction(Kt,etai,etat);
} // end refraction()

inline __host__ __device__ void refraction(const float3 &Kt, const float etai, const float etat, CudaScatteringDistributionFunction &f)
{
  return CudaShadingInterface::refraction(Kt,etai,etat,f);
} // end refraction()

