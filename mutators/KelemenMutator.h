/*! \file KelemenMutator.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a PathMutator
 *         implementing the simple, robust mutation
 *         scheme from [Kelemen, et al. 2002].
 */

#ifndef KELEMEN_MUTATOR_H
#define KELEMEN_MUTATOR_H

#include "PathMutator.h"
#include "../primitives/Scene.h"
#include "../path/PathSampler.h"
#include <boost/shared_ptr.hpp>

class KelemenMutator
  : public PathMutator
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef PathMutator Parent;

    /*! Null constructor calls the Parent.
     */
    KelemenMutator(void);

    /*! Constructor accepts a random sequence, a large step probability
     *  and a PathSampler.
     *  \param sequence Sets Parent::mRandomSequence.
     *  \param p Sets mLargeStepProbability.
     *  \param s Sets mSampler.
     */
    KelemenMutator(const boost::shared_ptr<RandomSequence> &sequence,
                   const float p,
                   const boost::shared_ptr<PathSampler> &s);

    /*! Constructor accepts a large step probability and a PathSampler.
     *  \param p Sets mLargeStepProbability.
     *  \param s Sets mSampler.
     */
    KelemenMutator(const float p,
                   const boost::shared_ptr<PathSampler> &s);

    virtual int mutate(const PathSampler::HyperPoint &x,
                       const Path &a,
                       PathSampler::HyperPoint &y,
                       Path &b);

    virtual float evaluateTransitionRatio(const unsigned int which,
                                          const PathSampler::HyperPoint &x,
                                          const Path &a,
                                          const float ix,
                                          const PathSampler::HyperPoint &y,
                                          const Path &b,
                                          const float iy) const;

    /*! This method sts mLargeStepProbability.
     *  \param p Sets mLargeStepProbability
     */
    void setLargeStepProbability(const float p);

    /*! This method returns mLargeStepProbability.
     *  \return mLargeStepProbability
     */
    float getLargeStepProbability(void) const;

    /*! This method generates a new HyperPoint with a large step.
     *  \param y The new HyperPoint is returned here.
     */
    virtual void largeStep(PathSampler::HyperPoint &y);

    /*! This method generates a new Path with a large step.
     *  \param y The new Path's HyperPoint is returned here.
     *  \param b The new Path is returned here.
     *  \return true if a new Path could be successfully created;
     *          false, otherwise.
     */
    virtual bool largeStep(PathSampler::HyperPoint &y,
                           Path &b);

    /*! This method mutates a Path into a new one with a small step.
     *  \param x The initial Path's HyperPoint.
     *  \param a The initial Path.
     *  \param y The result of applying a small step to x.
     *  \param b The result of applying a small step to a.
     *  \return true if x could be successfully mutated;
     *          false, othewise.
     */
    virtual bool smallStep(const PathSampler::HyperPoint &x,
                           const Path &a,
                           PathSampler::HyperPoint &y,
                           Path &b) const;

    /*! This method mutates a HyperPoint into a new one
     *  with a small step.
     *  \param x The initial HyperPoint.
     *  \param y The mutated HyperPoint is returned here.
     */
    virtual void smallStep(const PathSampler::HyperPoint &x,
                           PathSampler::HyperPoint &y) const;

    /*! This method mutates a number into a new one
     *  with a small step.
     *  \param x The initial number.
     *  \return The mutated number.
     */
    virtual float smallStep(const float x) const;

    virtual Spectrum evaluate(const Path &x,
                              std::vector<PathSampler::Result> &results);

  protected:
    /*! The probability of proposing a large step.
     */
    float mLargeStepProbability;
}; // end KelemenMutator

#endif // KELEMEN_MUTATOR_H

