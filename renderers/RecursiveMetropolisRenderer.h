/*! \file RecursiveMetropolisRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an implementation of
 *         recursive MLT.
 */

#ifndef RECURSIVE_METROPOLIS_RENDERER_H
#define RECURSIVE_METROPOLIS_RENDERER_H

#include "TargetRaysRenderer.h"

class RecursiveMetropolisRenderer
  : public TargetRaysRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef TargetRaysRenderer Parent;

    /*! Null constructor does nothing.
     */
    RecursiveMetropolisRenderer(void);

    /*! Constructor accepts a PathMutator and calls the
     *  Parent.
     *  \param s Sets Parent::mRandomSequence.
     *  \param m Sets mMutator.
     *  \param target Sets mRayTarget.
     */
    RecursiveMetropolisRenderer(const boost::shared_ptr<RandomSequence> &s,
                                const boost::shared_ptr<PathMutator> &mutator,
                                const unsigned int target);

    /*! Constructor accepts a pointer to a Scene, Film, and PathMutator.
     *  \param s Sets mScene.
     *  \param r Sets mRecord.
     *  \param s Sets Parent::mRandomSequence.
     *  \param m Sets mMutator.
     *  \param target Sets mRayTarget.
     */
    RecursiveMetropolisRenderer(boost::shared_ptr<const Scene> &s,
                                boost::shared_ptr<Record> &r,
                                const boost::shared_ptr<RandomSequence> &sequence,
                                const boost::shared_ptr<PathMutator> &m,
                                const unsigned int target);

    /*! This method coordinates preprocessing tasks prior to rendering.
     */
    virtual void preprocess(void);

  protected:
    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *                  called throughout the rendering process.
     */
    virtual void kernel(ProgressCallback &progress);

    // The recursion scale.
    float mRecursionScale;
}; // end RecursiveMetropolisRenderer

#endif // RECURSIVE_METROPOLIS_RENDERER_H

