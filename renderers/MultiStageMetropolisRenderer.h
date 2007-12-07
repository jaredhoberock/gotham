/*! \file MultiStageMetropolisRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a MetropolisRenderer which generates
 *         tentative estimates in stages and uses them to importance sample.
 */

#ifndef MULTI_STAGE_METROPOLIS_RENDERER_H
#define MULTI_STAGE_METROPOLIS_RENDERER_H

#include "TargetRaysRenderer.h"

class MultiStageMetropolisRenderer
  : public TargetRaysRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef TargetRaysRenderer Parent;

    /*! Null constructor does nothing.
     */
    MultiStageMetropolisRenderer(void);

    /*! Constructor accepts a pointer to a Scene, Film, and PathMutator.
     *  \param s Sets Parent::mRandomSequence.
     *  \param m Sets mMutator.
     *  \param importance Sets mImportance.
     *  \param target Sets mRayTarget.
     */
    MultiStageMetropolisRenderer(const boost::shared_ptr<RandomSequence> &sequence,
                                 const boost::shared_ptr<PathMutator> &m,
                                 const boost::shared_ptr<ScalarImportance> &importance,
                                 const unsigned int target);

  protected:
    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *                  called throughout the rendering process.
     */
    virtual void kernel(ProgressCallback &progress);

    // The recursion scale.
    float mRecursionScale;
}; // end MultiStageMetropolisRenderer

#endif // MULTI_STAGE_METROPOLIS_RENDERER_H

