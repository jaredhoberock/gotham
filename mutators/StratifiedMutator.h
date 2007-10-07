/*! \file StratifiedMutator.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a KelemenMutator
 *         whose large steps are more or less stratified
 *         over the image plane.
 */

#ifndef STRATIFIED_MUTATOR_H
#define STRATIFIED_MUTATOR_H

#include "KelemenMutator.h"
#include <gpcpu/Vector.h>

class StratifiedMutator
  : public KelemenMutator
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef KelemenMutator Parent;

    /*! Constructor accepts a value for large step probability.
     *  \param largeStepProbability Sets Parent::mLargeStepProbability.
     *  \param s Sets Parent::mSampler.
     *  \param strata Sets mStrata.
     */
    StratifiedMutator(const float largeStepProbability,
                      const boost::shared_ptr<PathSampler> &s,
                      const gpcpu::uint2 &strata);

    /*! This method generates a new HyperPoint with a large step.
     *  The HyperPoint's pixel coordinates are chosen from the list
     *  of stratified samples.
     *  \param y The new HyperPoint is returned here.
     */
    using Parent::largeStep;
    virtual void largeStep(PathSampler::HyperPoint &y);

  protected:
    /*! This method resets this StratifiedMutator's list of sample locations.
     */
    virtual void reset(void);

    std::vector<gpcpu::float2> mSampleLocations;
    unsigned int mCurrentSample;

    /*! The number of strata in either dimension.
     */
    gpcpu::uint2 mStrata;
}; // end StratifiedMutator

#endif // STRATIFIED_MUTATOR_H

