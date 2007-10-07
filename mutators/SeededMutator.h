/*! \file SeededMutator.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a KelemenMutator
 *         whose large steps return Paths known to be valid.
 */

#ifndef SEEDED_MUTATOR_H
#define SEEDED_MUTATOR_H

#include "KelemenMutator.h"
#include "../shading/FunctionAllocator.h"

class SeededMutator
  : public KelemenMutator
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef KelemenMutator Parent;

    /*! Null constructor calls the Parent.
     */
    SeededMutator(void);

    /*! Constructor accepts a large step probability,
     *  PathSampler, and the number of seeds to generate.
     *  \param p Sets Parent::mLargeStepProbability.
     *  \param s Sets mSampler.
     *  \param n The number of tries to generate seeds.
     */
    SeededMutator(const float p,
                  const boost::shared_ptr<PathSampler> &s,
                  const size_t n);

    /*! Constructor accepts a random sequence, a large step probability,
     *  PathSampler, and the number of seeds to generate.
     *  \param sequence Sets Parent::mRandomSequence.
     *  \param p Sets Parent::mLargeStepProbability.
     *  \param s Sets mSampler.
     *  \param n The number of tries to generate seeds.
     */
    SeededMutator(const boost::shared_ptr<RandomSequence> &sequence,
                  const float p,
                  const boost::shared_ptr<PathSampler> &s,
                  const size_t n);

    /*! This method generates a new Path with a large step.
     *  \param y The new Path's HyperPoint is returned here.
     *  \param b The new Path is returned here.
     *  \return true if a new Path could be successfully created;
     *          false, otherwise.
     */
    using Parent::largeStep;
    virtual bool largeStep(PathSampler::HyperPoint &y,
                           Path &b);

    /*! This method is called prior to rendering and generates the
     *  list of seeds.
     */
    virtual void preprocess(void);

    /*! This method is called after rendering and deletes the list of seeds.
     */
    virtual void postprocess(void);

  protected:
    /*! The number of rejection samples to take when generating seeds.
     */
    size_t mNumRejectionSamples;

    /*! A list of seed HyperPoints.
     */
    std::vector<PathSampler::HyperPoint> mSeedPoints;

    /*! A list of seeds.
     *  XXX We have to use pointers, since Path's copy constructor is private.
     *      This is incredibly annoying.
     */
    std::vector<Path*> mSeeds;

    /*! A local pool for our seeds' scattering functions
     */
    FunctionAllocator mLocalPool;
}; // end SeededMutator

#endif // SEEDED_MUTATOR_H

