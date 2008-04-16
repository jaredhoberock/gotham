/*! \file CudaShadingContext.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ShadingContext
 *         which can run shading operations in CUDA
 *         code.
 */

#pragma once

#include "CudaScatteringDistributionFunction.h"
#include "../../shading/ShadingContext.h"
#include "../include/CudaShadingInterface.h"
#include <stdcuda/vector_dev.h>

class CudaShadingContext
  : public ShadingContext,
    public CudaShadingInterface
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef ShadingContext Parent0;

    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef CudaShadingInterface Parent1;

    /*! This method evaluates a batch of scattering shader jobs in SIMD fashion
     *  on a CUDA device.
     *  \param m A list of MaterialHandles.
     *  \param dg A list of CudaDifferentialGeometry objects.
     *  \param dgStride Stride size for elements in the dg list.
     *  \param stencil A stencil to control job processing.
     *  \param f The results of shading operations will be returned to this list.
     *  \param n The size of each list.
     */
    using Parent0::evaluateScattering;
    virtual void evaluateScattering(const stdcuda::device_ptr<const MaterialHandle> &m,
                                    const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                                    const size_t dgStride,
                                    const stdcuda::device_ptr<const int> &stencil,
                                    const stdcuda::device_ptr<CudaScatteringDistributionFunction> &g,
                                    const size_t n) = 0;

    /*! This method evaluates a batch of emission shader jobs in SIMD fashion
     *  on a CUDA device.
     *  \param m A list of MaterialHandles.
     *  \param dg A list of CudaDifferentialGeometry objects.
     *  \param dgStride Stride size for elements in the dg list.
     *  \param stencil A stencil to control job processing.
     *  \param f The results of shading operations will be returned to this list.
     *  \param n The size of each list.
     */
    using Parent0::evaluateEmission;
    virtual void evaluateEmission(const stdcuda::device_ptr<const MaterialHandle> &m,
                                  const stdcuda::device_ptr<const CudaDifferentialGeometry> &dg,
                                  const size_t dgStride,
                                  const stdcuda::device_ptr<const int> &stencil,
                                  const stdcuda::device_ptr<CudaScatteringDistributionFunction> &g,
                                  const size_t n) = 0;


    /*! This method evaluates the bidirectional scattering of a batch of
     *  scattering jobs in a SIMD fashion on a CUDA device.
     *  \param f A list of ScatteringDistributionFunction objects.
     *  \param wo A list of outgoing directions.
     *  \param dg A list of DifferentialGeometry objects.
     *  \param wi A list of incoming directions.
     *  \param stencil This stencil controls which jobs in the list get processed.
     *  \param results The result of each scattering job is returned to this list.
     *  \param n The number of jobs.
     */
    virtual void evaluateBidirectionalScattering(ScatteringDistributionFunction **f,
                                                 const Vector *wo,
                                                 const DifferentialGeometry *dg,
                                                 const Vector *wi,
                                                 const int *stencil,
                                                 Spectrum *results,
                                                 const size_t n);

    virtual void evaluateBidirectionalScattering(stdcuda::device_ptr<const CudaScatteringDistributionFunction> f,
                                                 stdcuda::device_ptr<const float3> wo,
                                                 stdcuda::device_ptr<const CudaDifferentialGeometry> dg,
                                                 stdcuda::device_ptr<const float3> wi,
                                                 stdcuda::device_ptr<const int> stencil,
                                                 stdcuda::device_ptr<float3> results,
                                                 const size_t n);

    virtual void evaluateBidirectionalScattering(stdcuda::device_ptr<const CudaScatteringDistributionFunction> f,
                                                 stdcuda::device_ptr<const float3> wo,
                                                 stdcuda::device_ptr<const CudaDifferentialGeometry> dg,
                                                 const size_t dgStride,
                                                 stdcuda::device_ptr<const float3> wi,
                                                 stdcuda::device_ptr<const int> stencil,
                                                 stdcuda::device_ptr<float3> results,
                                                 const size_t n);

    /*! This method evaluates the unidirectional scattering of a batch of
     *  scattering jobs in a SIMD fashion on a CUDA device.
     *  \param f A list of ScatteringDistributionFunction objects.
     *  \param wo A list of outgoing directions.
     *  \param dg A list of DifferentialGeometry objects.
     *  \param stencil This stencil controls which jobs in the list get processed.
     *  \param results The result of each scattering job is returned to this list.
     *  \param n The number of jobs.
     */
    virtual void evaluateUnidirectionalScattering(ScatteringDistributionFunction **f,
                                                  const Vector *wo,
                                                  const DifferentialGeometry *dg,
                                                  const int *stencil,
                                                  Spectrum *results,
                                                  const size_t n);

    virtual void evaluateUnidirectionalScattering(stdcuda::device_ptr<const CudaScatteringDistributionFunction> f,
                                                  stdcuda::device_ptr<const float3> wo,
                                                  stdcuda::device_ptr<const CudaDifferentialGeometry> dg,
                                                  stdcuda::device_ptr<const int> stencil,
                                                  stdcuda::device_ptr<float3> results,
                                                  const size_t n);

    virtual void evaluateUnidirectionalScattering(stdcuda::device_ptr<const CudaScatteringDistributionFunction> f,
                                                  stdcuda::device_ptr<const float3> wo,
                                                  stdcuda::device_ptr<const CudaDifferentialGeometry> dg,
                                                  const size_t dgStride,
                                                  stdcuda::device_ptr<const int> stencil,
                                                  stdcuda::device_ptr<float3> results,
                                                  const size_t n);

    /*! This method creates a list of CudaScatteringDistributionFunctions given
     *  a list of ScatteringDistributionFunctions.
     *  \param f The input list of ScatteringDistributionFunctions.
     *  \param stencil A stencil to control which jobs get processed.
     *  \param n The size of f.
     *  \param cf The output list of CudaScatteringDistributionFunctions is returned here.
     */
    virtual void createCudaScatteringDistributionFunctions(ScatteringDistributionFunction **f,
                                                           const int *stencil,
                                                           const size_t n,
                                                           stdcuda::vector_dev<CudaScatteringDistributionFunction> &cf);

    /*! This method copies the CudaMaterials from the given MaterialList into this
     *  CudaShadingContext's MaterialList. If a Material in the given list is not a
     *  CudaMaterial, a default one is created in its place.
     *  \param materials The list to copy.
     */
    virtual void setMaterials(const boost::shared_ptr<MaterialList> &materials);

  protected:

    virtual CudaScatteringDistributionFunction createCudaScatteringDistributionFunction(const ScatteringDistributionFunction *f);
}; // end CudaShadingContext

