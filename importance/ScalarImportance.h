/*! \file ScalarImportance.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         which assigns scalar importance to Paths
 *         produced by MLT.
 */

#ifndef SCALAR_IMPORTANCE_H
#define SCALAR_IMPORTANCE_H

#include "../path/PathSampler.h"
#include "../mutators/PathMutator.h"

class MetropolisRenderer;
class FunctionAllocator;

class ScalarImportance
{
  public:
    /*! This method is called prior to rendering.
     *  \param r A sequence of RandomNumbers.
     *  \param scene The Scene to be rendered.
     *  \param mutator The PathMutator to be used duing the rendering process.
     *  \param renderer A reference to the MetropolisRenderer owning this
     *                  ScalarImportance.
     */
    virtual void preprocess(const boost::shared_ptr<RandomSequence> &r,
                            const boost::shared_ptr<const Scene> &scene,
                            const boost::shared_ptr<PathMutator> &mutator,
                            MetropolisRenderer &renderer);

    /*! This method returns mNormalizationConstant.
     *  \return mNormalizationConstant.
     */
    inline float getNormalizationConstant(void) const;

    /*! This method returns mInvNormalizationConstant.
     *  \return mInvNormalizationConstant
     */
    inline float getInvNormalizationConstant(void) const;

    /*! This method calls evaluate().
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param xPath The Path corresponding to x.
     *  \param results The list of PathSampler Results resulting from xPath.
     *  \return evaluate(x,f)
     */
    float operator()(const PathSampler::HyperPoint &x,
                     const Path &xPath,
                     const std::vector<PathSampler::Result> &results);

    /*! This method assigns a scalar importance to a Path.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param xPath The Path corresponding to x.
     *  \param results The list of PathSampler Results resulting from xPath.
     *  \return The scalar importance of x.
     *  \note This method must be implemented in a derived class.
     */
    virtual float evaluate(const PathSampler::HyperPoint &x,
                           const Path &xPath,
                           const std::vector<PathSampler::Result> &results) = 0;

    /*! This method estimates the normalization constant of this ScalarImportance
     *  function and chooses a seed Path proportional to its importance.
     *  \param r A sequence of RandomNumbers.
     *  \param scene The Scene to be rendered.
     *  \param mutator The PathMutator to be used duing the rendering process.
     *  \param n The number of samples to use in the estimation.
     *  \param allocator A FunctionAllocator to use when allocating scattering functions
     *                   in the returned Path.
     *  \param x The HyperPoint of the chosen seed is returned here.
     *  \param xPath The chosen seed Path is returned here.
     *  \return The estimate of the normalization constant.
     *  XXX DESIGN I don't like that this is public, but we have the requirement that we need
     *             a seed distributed porportional to importance.
     */
    virtual float estimateNormalizationConstant(const boost::shared_ptr<RandomSequence> &r,
                                                const boost::shared_ptr<const Scene> &scene,
                                                const boost::shared_ptr<PathMutator> &mutator,
                                                const size_t n,
                                                FunctionAllocator &allocator,
                                                PathSampler::HyperPoint &x,
                                                Path &xPath);

    /*! This method estimates the normalization constant of this ScalarImportance
     *  function but does not choose a seed Path proportional to its importance.
     *  \param r A sequence of RandomNumbers.
     *  \param scene The Scene to be rendered.
     *  \param mutator The PathMutator to be used duing the rendering process.
     *  \param n The number of samples to use in the estimation.
     *  \return The estimate of the normalization constant.
     */
    virtual float estimateNormalizationConstant(const boost::shared_ptr<RandomSequence> &r,
                                                const boost::shared_ptr<const Scene> &scene,
                                                const boost::shared_ptr<PathMutator> &mutator,
                                                const size_t n);

  protected:
    /*! The normalization constant of this ScalarImportance function.
     */
    float mNormalizationConstant;

    /*! The reciprocal of mNormalizationConstant.
     */
    float mInvNormalizationConstant;
}; // end ScalarImportance

#include "ScalarImportance.inl"

#endif // SCALAR_IMPORTANCE_H

