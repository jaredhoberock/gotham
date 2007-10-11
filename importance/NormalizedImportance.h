/*! \file NormalizedImportance.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScalarImportance
 *         class which normalizes importance given an
 *         intial neighborhood estimate of importance.
 */

#ifndef NORMALIZED_IMPORTANCE_H
#define NORMALIZED_IMPORTANCE_H

#include "ScalarImportance.h"
#include "../numeric/RandomSequence.h"
#include "../mutators/PathMutator.h"
#include <array2/Array2.h>

class NormalizedImportance
  : public ScalarImportance
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScalarImportance Parent;

    /*! This method is to be called prior to rendering.
     *  \param r A sequence of random numbers.
     *  \param scene The Scene to be rendered.
     *  \param mutator The PathMutator to be used in the rendering process.
     */
    virtual void preprocess(const boost::shared_ptr<RandomSequence> &r,
                            const boost::shared_ptr<const Scene> &scene,
                            const boost::shared_ptr<PathMutator> &mutator,
                            MetropolisRenderer &renderer);

    /*! This method converts the spectral Monte Carlo throughput
     *  of a Path into scalar importance.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param f The spectral Monte Carlo throughput of the Path of interest.
     *  \return The normalized scalar importance of x.
     */
    virtual float evaluate(const PathSampler::HyperPoint &x,
                           const Path &xPath,
                           const std::vector<PathSampler::Result> &results);

  protected:
    /*! A low resolution estimate of the image
     *  to be rendered.
     */
    Array2<float> mEstimate;
}; // end NormalizedImportance

#endif // NORMALIZED_IMPORTANCE_H

