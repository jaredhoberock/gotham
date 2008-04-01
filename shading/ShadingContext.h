/*! \file ShadingContext.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to the api used by shaders to
 *         create ScatteringDistributionFunctions.
 */

#pragma once

#include "../include/exportShading.h"
#include "../include/ShadingInterface.h"

#ifdef WIN32
class DLLAPI ShadingContext;
#endif // WIN32

#include "FunctionAllocator.h"
#include "MaterialList.h"
#include <boost/shared_ptr.hpp>

class ShadingContext
  : public ShadingInterface
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ShadingInterface Parent;

    /*! Null destructor does nothing.
     */
    virtual ~ShadingContext(void);

    /*! Constructor accepts a list of Materials.
     *  \param materials Sets mMaterials.
     */
    ShadingContext(const boost::shared_ptr<MaterialList> &materials);

    ScatteringDistributionFunction *null(void);

    ScatteringDistributionFunction *diffuse(const Spectrum &Kd);

    ScatteringDistributionFunction *glossy(const Spectrum &Kr,
                                           const float eta,
                                           float exponent);

    ScatteringDistributionFunction *glossy(const Spectrum &Kr,
                                           const float eta,
                                           float uExponent,
                                           float vExponent);

    ScatteringDistributionFunction *glossyRefraction(const Spectrum &Kt,
                                                     const float etai,
                                                     const float etat,
                                                     const float exponent);


    ScatteringDistributionFunction *glass(const float eta,
                                          const Spectrum &Kr,
                                          const Spectrum &Kt);

    ScatteringDistributionFunction *thinGlass(const float eta,
                                              const Spectrum &Kr,
                                              const Spectrum &Kt);

    ScatteringDistributionFunction *mirror(const Spectrum &Kr,
                                           const float eta);

    ScatteringDistributionFunction *refraction(const Spectrum &Kt,
                                               const float etai,
                                               const float etat);

    ScatteringDistributionFunction *transparent(const Spectrum &Kt);

    ScatteringDistributionFunction *uber(const Spectrum &Kd,
                                         const Spectrum &Ks,
                                         const float uShininess,
                                         const float vShininess);

    ScatteringDistributionFunction *uber(const Spectrum &Kd,
                                         const Spectrum &Ks,
                                         const float shininess);

    ScatteringDistributionFunction *uber(const Spectrum &Ks,
                                         const float shininess,
                                         const Spectrum &Kr,
                                         float eta);

    ScatteringDistributionFunction *perspectiveSensor(const Spectrum &Ks,
                                                      const float aspect,
                                                      const Point &origin,
                                                      const Vector &right,
                                                      const Vector &up);

    ScatteringDistributionFunction *hemisphericalEmission(const Spectrum &Ke);

    float noise(const float x, const float y, const float z);

    /*! This method frees all allocated shading resources.
     */
    virtual void freeAll(void);

    /*! This method clones a ScatteringDistributionFunction, possibly resident
     *  in a different ShadingContext, and makes the clone a resident this ShadingContext.
     *  \param s The ScatteringDistributionFunction of interest.
     *  \return A clone of s, which is a resident of this ShadingContext.
     */
    virtual ScatteringDistributionFunction *clone(const ScatteringDistributionFunction *s);

    /*! This method returns a reference to mAllocator.
     *  \return mAllocator
     */
    FunctionAllocator &getAllocator(void);

    /*! This method evaluates a scattering shader.
     *  \param m The Material of interest.
     *  \param dg The DifferentialGeometry at the shading point of interest.
     *  \return A pointer to a newly allocated ScatteringDistributionFunction at the shading point of interest.
     */
    virtual ScatteringDistributionFunction *evaluateScattering(const MaterialHandle &m,
                                                               const DifferentialGeometry &dg);

    /*! This method evaluates the scattering of a list of shading points in a SIMD fashion.
     *  \param m A list of Materials to evaluate.
     *  \param dg A list of shading points of interest.
     *  \param stencil This stencil controls which jobs in the list get processed.
     *  \param f Newly allocated ScatteringDistributionFunctions are returned to this
     *           list corresponding to each shading job.
     *  \param n The number of jobs.
     */
    virtual void evaluateScattering(const MaterialHandle *m,
                                    const DifferentialGeometry *dg,
                                    const int *stencil,
                                    ScatteringDistributionFunction **f,
                                    const size_t n);

    /*! This method evaluates a sensor shader.
     *  \param m The Material of interest.
     *  \param dg The DifferentialGeometry at the shading point of interest.
     *  \return A pointer to a newly allocated ScatteringDistributionFunction at the shading point of interest.
     */
    virtual ScatteringDistributionFunction *evaluateSensor(const MaterialHandle &m,
                                                           const DifferentialGeometry &dg);

    /*! This method evaluates the sensing of a list of shading points in a SIMD fashion.
     *  \param m A list of Materials to evaluate.
     *  \param dg A list of shading points of interest.
     *  \param stencil This stencil controls which jobs in the list get processed.
     *  \param f Newly allocated ScatteringDistributionFunctions are returned to this
     *           list corresponding to each shading job.
     *  \param n The number of jobs.
     */
    virtual void evaluateSensor(const MaterialHandle *m,
                                const DifferentialGeometry *dg,
                                const int *stencil,
                                ScatteringDistributionFunction **f,
                                const size_t n);

    /*! This method evaluates an emission shader.
     *  \param m The Material of interest.
     *  \param dg The DifferentialGeometry at the shading point of interest.
     *  \return A pointer to a newly allocated ScatteringDistributionFunction at the shading point of interest.
     */
    virtual ScatteringDistributionFunction *evaluateEmission(const MaterialHandle &m,
                                                             const DifferentialGeometry &dg);

    /*! This method evaluates the emission of a list of shading points in a SIMD fashion.
     *  \param m A list of Materials to evaluate.
     *  \param dg A list of shading points of interest.
     *  \param stencil This stencil controls which jobs in the list get processed.
     *  \param f Newly allocated ScatteringDistributionFunctions are returned to this
     *           list corresponding to each shading job.
     *  \param n The number of jobs.
     */
    virtual void evaluateEmission(const MaterialHandle *m,
                                  const DifferentialGeometry *dg,
                                  const int *stencil,
                                  ScatteringDistributionFunction **f,
                                  const size_t n);

    /*! This method evaluates the bidirectional scattering of a batch of
     *  scattering jobs in a SIMD fashion.
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
     *  scattering jobs in SIMD fashion.
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
    FunctionAllocator mAllocator;

    /*! A ShadingContext keeps a list of Materials.
     */
    boost::shared_ptr<MaterialList> mMaterials;
}; // end ShadingContext

