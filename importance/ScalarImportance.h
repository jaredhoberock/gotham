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

class ScalarImportance
{
  public:
    /*! This method is called prior to rendering.
     *  \param r A sequence of RandomNumbers.
     *  \param scene The Scene to be rendered.
     *  \param mutator The PathMutator to be used duing the rendering process.
     *  \param renderer A reference to the MetropolisRenderer owning this
     *                  ScalarImportance.
     *  \note The default implementation of this method does nothing.
     */
    virtual void preprocess(const boost::shared_ptr<RandomSequence> &r,
                            const boost::shared_ptr<const Scene> &scene,
                            const boost::shared_ptr<PathMutator> &mutator,
                            MetropolisRenderer &renderer);

    /*! This method converts the spectral Monte Carlo throughput
     *  of a Path into scalar importance.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param f The spectral Monte Carlo throughput of the Path of interest.
     *  \return The scalar importance of x.
     *  \note This method must be implemented in a derived class.
     */
    virtual float operator()(const PathSampler::HyperPoint &x,
                             const Spectrum &f) = 0;
}; // end ScalarImportance

#endif // SCALAR_IMPORTANCE_H

