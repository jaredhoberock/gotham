/*! \file CudaShadingContext.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ShadingContext
 *         which can run shading operations in CUDA
 *         code.
 */

#pragma once

#include "../../shading/ShadingContext.h"
#include "CudaScatteringDistributionFunction.h"
#include <stdcuda/vector_dev.h>

class CudaShadingContext
  : public ShadingContext
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ShadingContext Parent;

    /*! Constructor accepts a list of materials.
     *  \param materials Sets Parent::mMaterials.
     */
    CudaShadingContext(const boost::shared_ptr<MaterialList> &materials);

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

  protected:
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

    virtual CudaScatteringDistributionFunction createCudaScatteringDistributionFunction(const ScatteringDistributionFunction *f);
}; // end CudaShadingContext

