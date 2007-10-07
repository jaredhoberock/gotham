/*! \file EnergyRedistributionRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a MetropolisRenderer
 *         implementing Cline, Talbot, & Egbert, 05.
 */

#ifndef ENERGY_REDISTRIBUTION_RENDERER_H
#define ENERGY_REDISTRIBUTION_RENDERER_H

#include "MetropolisRenderer.h"

class EnergyRedistributionRenderer
  : public MetropolisRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef MetropolisRenderer Parent;

    /*! Constructor accepts a PathMutator and calls the
     *  Parent.
     *  \param mutationsPerSample Sets mMutationsPerSample.
     *  \param chainLength Sets mChainLength.
     *  \param s Sets Parent::mRandomSequence.
     *  \param m Sets mMutator.
     *  \param i Sets mImportance.
     */
    EnergyRedistributionRenderer(const float mutationsPerSample,
                                 const unsigned int chainLength,
                                 const boost::shared_ptr<RandomSequence> &s,
                                 const boost::shared_ptr<PathMutator> &mutator,
                                 const boost::shared_ptr<ScalarImportance> &importance);

  protected:
    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *                  called throughout the rendering process.
     */
    virtual void kernel(ProgressCallback &progress);

    /*! The desired number of mutations per Monte Carlo sample.
     */
    float mMutationsPerSample;

    /*! The chain length.
     */
    unsigned int mChainLength;
}; // end EnergyRedistributionRenderer

#endif // ENERGY_REDISTRIBUTION_RENDERER_H

