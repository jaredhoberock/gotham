/*! \file TargetRaysRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a MetropolisRenderer
 *         which stops after achieving a target number of Rays cast.
 *  XXX DESIGN There should be a way for MonteCarloRenderer to do the same
 *             thing; there's really no reason to for a whole new Renderer class
 *             to be made for this feature.
 */

#ifndef TARGET_RAYS_RENDERER_H
#define TARGET_RAYS_RENDERER_H

#include "MetropolisRenderer.h"

class TargetRaysRenderer
  : public MetropolisRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef MetropolisRenderer Parent;

    /*! Null constructor does nothing.
     */
    TargetRaysRenderer(void);

    /*! Constructor accepts a PathMutator and calls the
     *  null constructor of the Parent.
     *  \param s Sets Parent::mRandomSequence.
     *  \param m Sets mMutator.
     *  \param i Sets mImportance.
     *  \param target Sets mRayTarget.
     */
    TargetRaysRenderer(const boost::shared_ptr<RandomSequence> &s,
                       const boost::shared_ptr<PathMutator> &mutator,
                       const boost::shared_ptr<ScalarImportance> &importance,
                       const unsigned int target);

    /*! Constructor accepts a pointer to a Scene, Film, and PathMutator.
     *  \param s Sets mScene.
     *  \param f Sets mFilm.
     *  \param s Sets Parent::mRandomSequence.
     *  \param m Sets mMutator.
     *  \param i Sets mImportance.
     *  \param target Sets mRayTarget.
     */
    TargetRaysRenderer(boost::shared_ptr<const Scene> &s,
                       boost::shared_ptr<RenderFilm> &f,
                       const boost::shared_ptr<RandomSequence> &sequence,
                       const boost::shared_ptr<PathMutator> &m,
                       const boost::shared_ptr<ScalarImportance> &i,
                       const unsigned int target);

  protected:
    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *                  called throughout the rendering process.
     */
    virtual void kernel(ProgressCallback &progress);

    /*! This method is called after kernel().
     */
    virtual void postprocess(void);

    /*! A loose limit on the number of rays to cast.
     */
    unsigned int mRayTarget;

    /*! The number of samples taken.
     *  XXX DESIGN: move this somewhere common
     */
    unsigned int mNumSamplesTaken;
}; // end TargetRaysRenderer

#endif // TARGET_RAYS_RENDERER_H

