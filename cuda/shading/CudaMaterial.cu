/*! \file CudaMaterial.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaMaterial class.
 */

#define inline inline __host__ __device__

#include "../include/CudaMaterial.h"
#include "../include/CudaShadingInterface.h"
#include <stdcuda/fill_if.h>
using namespace stdcuda;

#undef inline

void CudaMaterial
  ::evaluateScattering(CudaShadingInterface &context,
                       const device_ptr<const CudaDifferentialGeometry> &dg,
                       const device_ptr<const bool> &stencil,
                       const device_ptr<CudaScatteringDistributionFunction> &f,
                       const size_t n) const
{
  CudaScatteringDistributionFunction value;
  context.null(value);
  fill_if(f.get(), f.get() + n, stencil.get(), value);
} // end CudaMaterial::evaluateScattering()

void CudaMaterial
  ::evaluateScattering(CudaShadingInterface &context,
                       const device_ptr<const CudaDifferentialGeometry> &dg,
                       const size_t dgStride,
                       const device_ptr<const bool> &stencil,
                       const device_ptr<CudaScatteringDistributionFunction> &f,
                       const size_t n) const
{
  CudaScatteringDistributionFunction value;
  context.null(value);
  fill_if(f.get(), f.get() + n, stencil.get(), value);
} // end CudaMaterial::evaluateScattering()

void CudaMaterial
  ::evaluateEmission(CudaShadingInterface &context,
                     const device_ptr<const CudaDifferentialGeometry> &dg,
                     const size_t dgStride,
                     const device_ptr<const bool> &stencil,
                     const device_ptr<CudaScatteringDistributionFunction> &f,
                     const size_t n) const
{
  CudaScatteringDistributionFunction value;
  context.null(value);
  fill_if(f.get(), f.get() + n, stencil.get(), value);
} // end CudaMaterial::evaluateEmission()

void CudaMaterial
  ::evaluateSensor(CudaShadingInterface &context,
                   const device_ptr<const CudaDifferentialGeometry> &dg,
                   const size_t dgStride,
                   const device_ptr<const bool> &stencil,
                   const device_ptr<CudaScatteringDistributionFunction> &f,
                   const size_t n) const
{
  CudaScatteringDistributionFunction value;
  context.null(value);
  fill_if(f.get(), f.get() + n, stencil.get(), value);
} // end CudaMaterial::evaluateSensor()

