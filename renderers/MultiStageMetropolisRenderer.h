/*! \file MultiStageMetropolisRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a MetropolisRenderer which generates
 *         tentative estimates in stages and uses them to importance sample.
 */

#ifndef MULTI_STAGE_METROPOLIS_RENDERER_H
#define MULTI_STAGE_METROPOLIS_RENDERER_H

#include "MetropolisRenderer.h"

class MultiStageMetropolisRenderer
  : public MetropolisRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef MetropolisRenderer Parent;

    /*! Null constructor does nothing.
     */
    MultiStageMetropolisRenderer(void);

    /*! Constructor accepts a pointer to a Scene, Film, and PathMutator.
     *  \param s Sets Parent::mRandomSequence.
     *  \param m Sets mMutator.
     *  \param importance Sets mImportance.
     */
    MultiStageMetropolisRenderer(const boost::shared_ptr<RandomSequence> &sequence,
                                 const boost::shared_ptr<PathMutator> &m,
                                 const boost::shared_ptr<ScalarImportance> &importance);

  protected:
    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *                  called throughout the rendering process.
     */
    virtual void kernel(ProgressCallback &progress);

    /*! This method is periodically called during kernel() to update
     *  mImportance.
     *  \param bLuminance The mean pixel value of the resulting rendered image.
     *         XXX This should be available as a member of MetropolisRenderers
     *             in general.
     *  \param w The width of the new importance image.
     *  \param h The height of the new importance image.
     *  \param x The current state of x.
     *  \param xPath The current state of xPath.
     *  \param xResults The current state of xResults.
     *  \param ix x's importance will be updated here to reflect the updated importance function.
     *  \param xPdf x's pdf will be updated here to reflect the updated importance function.
     *  \return The reciprocal of the normalization constant of the updated importance function.
     */
    virtual float updateImportance(const float bLuminance,
                                   const float w,
                                   const float h,
                                   const PathSampler::HyperPoint &x,
                                   const Path &xPath,
                                   const std::vector<PathSampler::Result> &xResults,
                                   float &ix,
                                   float &xPdf);


    // The recursion scale.
    float mRecursionScale;

    FunctionAllocator mSeedPool;
    std::vector<PathSampler::HyperPoint> mSeedPoints;
    std::vector<Path> mSeedPaths;
    std::vector<PathSampler::ResultList> mSeedResults;
}; // end MultiStageMetropolisRenderer

#endif // MULTI_STAGE_METROPOLIS_RENDERER_H

