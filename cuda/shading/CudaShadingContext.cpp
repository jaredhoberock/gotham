/*! \file CudaShadingContext.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaShadingContext class.
 */

#include "CudaShadingContext.h"
#include "CudaScatteringDistributionFunction.h"
#include "CudaLambertian.h"
#include "CudaHemisphericalEmission.h"
#include "../include/CudaMaterial.h"
#include "CudaDefaultMaterial.h"
#include "../../shading/Lambertian.h"
#include "../../shading/HemisphericalEmission.h"
#include "evaluateBidirectionalScattering.h"
#include "evaluateUnidirectionalScattering.h"
#include <stdcuda/cuda_algorithm.h>
#include <vector_functions.h>

using namespace stdcuda;

void CudaShadingContext
  ::setMaterials(const boost::shared_ptr<MaterialList> &materials)
{
  mMaterials.reset(new MaterialList());

  // go through the Materials; only add one if it is a CudaMaterial
  // if it is not, create a CudaMaterial in its stead
  for(MaterialList::iterator m = materials->begin();
      m != materials->end();
      ++m)
  {
    if(dynamic_cast<CudaMaterial*>(m->get()) || (*m)->isSensor() || (*m)->isEmitter())
    {
      mMaterials->push_back(*m);
    } // end if
    else
    {
      mMaterials->push_back(boost::shared_ptr<Material>(new CudaDefaultMaterial()));
    } // end else
  } // end for i
} // end CudaShadingContext::setMaterials()

void CudaShadingContext
  ::evaluateBidirectionalScattering(device_ptr<const CudaScatteringDistributionFunction> f,
                                    device_ptr<const float3> wo,
                                    device_ptr<const CudaDifferentialGeometry> dg,
                                    device_ptr<const float3> wi,
                                    device_ptr<const bool> stencil,
                                    device_ptr<float3> results,
                                    const size_t n)
{
  // just pass to the kernel
  ::evaluateBidirectionalScattering(f, wo, dg, wi, stencil, results, n);
} // end CudaShadingContext::evaluateBidirectionalScattering()

void CudaShadingContext
  ::evaluateBidirectionalScattering(device_ptr<const CudaScatteringDistributionFunction> f,
                                    device_ptr<const float3> wo,
                                    device_ptr<const CudaDifferentialGeometry> dg,
                                    const size_t dgStride,
                                    device_ptr<const float3> wi,
                                    device_ptr<const bool> stencil,
                                    device_ptr<float3> results,
                                    const size_t n)
{
  // just pass to the kernel
  ::evaluateBidirectionalScatteringStride(f, wo, dg, dgStride, wi, stencil, results, n);
} // end CudaShadingContext::evaluateBidirectionalScattering()

void CudaShadingContext
  ::evaluateBidirectionalScattering(ScatteringDistributionFunction **f,
                                    const Vector *wo,
                                    const DifferentialGeometry *dg,
                                    const Vector *wi,
                                    const bool *stencil,
                                    Spectrum *results,
                                    const size_t n)
{
  // create CUDA-compatible bsdfs
  stdcuda::vector_dev<CudaScatteringDistributionFunction> cf;
  createCudaScatteringDistributionFunctions(f, stencil, n, cf);

  // create device scratch space
  stdcuda::vector_dev<float3> woDevice(n);
  stdcuda::vector_dev<float3> wiDevice(n);
  stdcuda::vector_dev<CudaDifferentialGeometry> dgDevice(n);
  stdcuda::vector_dev<bool> stencilDevice(n);
  stdcuda::copy(wo, wo + n, woDevice.begin());
  stdcuda::copy(wi, wi + n, wiDevice.begin());
  stdcuda::copy(dg, dg + n, dgDevice.begin());
  stdcuda::copy(stencil, stencil + n, stencilDevice.begin());

  stdcuda::vector_dev<float3> resultsDevice(n);
  evaluateBidirectionalScattering(&cf[0],
                                  &woDevice[0],
                                  &dgDevice[0],
                                  &wiDevice[0],
                                  &stencilDevice[0],
                                  &resultsDevice[0],
                                  n);

  // copy results back to host
  // XXX this is a bit of a hack, but it will work
  float3 *hostPtr = reinterpret_cast<float3*>(&results[0]);
  stdcuda::copy(resultsDevice.begin(), resultsDevice.end(), hostPtr);
} // end CudaShadingContext::evaluateBidirectionalScattering()

void CudaShadingContext
  ::evaluateUnidirectionalScattering(device_ptr<const CudaScatteringDistributionFunction> f,
                                     device_ptr<const float3> wo,
                                     device_ptr<const CudaDifferentialGeometry> dg,
                                     device_ptr<const bool> stencil,
                                     device_ptr<float3> results,
                                     const size_t n)
{
  // just pass to the kernel
  ::evaluateUnidirectionalScattering(f, wo, dg, stencil, results, n);
} // end CudaShadingContext::evaluateUnidirectionalScattering()

void CudaShadingContext
  ::evaluateUnidirectionalScattering(device_ptr<const CudaScatteringDistributionFunction> f,
                                     device_ptr<const float3> wo,
                                     device_ptr<const CudaDifferentialGeometry> dg,
                                     const size_t dgStride,
                                     device_ptr<const bool> stencil,
                                     device_ptr<float3> results,
                                     const size_t n)
{
  // just pass to the kernel
  ::evaluateUnidirectionalScatteringStride(f, wo, dg, dgStride, stencil, results, n);
} // end CudaShadingContext::evaluateUnidirectionalScattering()

void CudaShadingContext
  ::evaluateUnidirectionalScattering(ScatteringDistributionFunction **f,
                                     const Vector *wo,
                                     const DifferentialGeometry *dg,
                                     const bool *stencil,
                                     Spectrum *results,
                                     const size_t n)
{
  // create CUDA-compatible bsdfs
  stdcuda::vector_dev<CudaScatteringDistributionFunction> cf;
  createCudaScatteringDistributionFunctions(f, stencil, n, cf);

  // create device scratch space
  stdcuda::vector_dev<float3> woDevice(n);
  stdcuda::vector_dev<CudaDifferentialGeometry> dgDevice(n);
  stdcuda::vector_dev<bool> stencilDevice(n);
  stdcuda::copy(wo, wo + n, woDevice.begin());
  stdcuda::copy(dg, dg + n, dgDevice.begin());
  stdcuda::copy(stencil, stencil + n, stencilDevice.begin());

  stdcuda::vector_dev<float3> resultsDevice(n);
  evaluateUnidirectionalScattering(&cf[0],
                                   &woDevice[0],
                                   &dgDevice[0],
                                   &stencilDevice[0],
                                   &resultsDevice[0],
                                   n);

  // copy results back to host
  stdcuda::copy(resultsDevice.begin(), resultsDevice.end(), &results[0]);
} // end CudaShadingContext::evaluateUnidirectionalScattering()

void CudaShadingContext
  ::createCudaScatteringDistributionFunctions(ScatteringDistributionFunction **f,
                                              const bool *stencil,
                                              const size_t n,
                                              stdcuda::vector_dev<CudaScatteringDistributionFunction> &cf)
{
  cf.resize(n);
  for(size_t i = 0; i != n; ++i)
  {
    if(stencil[i])
    {
      cf[i] = createCudaScatteringDistributionFunction(f[i]);
    } // end if
  } // end for i
} // end CudaScatteringDistributionFunctions::createCudaScatteringDistributionFunctions()

CudaScatteringDistributionFunction CudaShadingContext
  ::createCudaScatteringDistributionFunction(const ScatteringDistributionFunction *f)
{
  CudaScatteringDistributionFunction result;

  // switch on the type of f
  if(dynamic_cast<const Lambertian*>(f) != 0)
  {
    const Lambertian *l = static_cast<const Lambertian*>(f);

    // XXX it's a shame we can't just do a straight copy
    Spectrum a = l->getAlbedo();
    float3 albedo = make_float3(a[0],a[1],a[2]);

    // create a diffuse scattering object
    Parent1::diffuse(albedo, result);
  } // end if
  else if(dynamic_cast<const HemisphericalEmission*>(f) != 0)
  {
    const HemisphericalEmission *he = static_cast<const HemisphericalEmission*>(f);

    // XXX it's a shame we can't just do a straight copy
    Spectrum r = he->getRadiosity();
    float3 radiosity = make_float3(r[0],r[1],r[2]);

    // create a hemispherical emission object
    Parent1::hemisphericalEmission(radiosity, result);
  } // end else if
  else
  {
    // create a null scattering object
    Parent1::null(result);
  } // end else

  return result;
} // end CudaShadingContext::createCudaScatteringDistributionFunction()

