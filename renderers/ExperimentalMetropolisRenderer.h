/*! \file ExperimentalMetropolisRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Renderer
 *         implementing Metropolis light transport.
 */

#ifndef EXPERIMENTAL_METROPOLIS_RENDERER_H
#define EXPERIMENTAL_METROPOLIS_RENDERER_H

#include "MetropolisRenderer.h"

class ExperimentalMetropolisRenderer
  : public MetropolisRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef MetropolisRenderer Parent;

    /*! Null constructor does nothing.
     */
    ExperimentalMetropolisRenderer(void);

    /*! Constructor accepts a PathMutator and calls the
     *  null constructor of the Parent.
     *  \param s Sets Parent::mRandomSequence.
     *  \param m Sets mMutator.
     *  \param i Sets mImportance.
     */
    ExperimentalMetropolisRenderer(const boost::shared_ptr<RandomSequence> &s,
                                   const boost::shared_ptr<PathMutator> &mutator,
                                   const boost::shared_ptr<ScalarImportance> &importance);

  protected:
    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *                  called throughout the rendering process.
     */
    virtual void kernel(ProgressCallback &progress);

    /*! This counter counts the number of consecutive rejections.
     */
    float mConsecutiveRejections;
}; // end MetropolisRenderer

#endif // METROPOLIS_RENDERER_H

