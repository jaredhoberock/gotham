/*! \file NoiseAwareMetropolisRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a MultiStageMetropolisRenderer
 *         which attempts to sample proportional to noise.
 */

#ifndef NOISE_AWARE_METROPOLIS_RENDERER_H
#define NOISE_AWARE_METROPOLIS_RENDERER_H

#include "MultiStageMetropolisRenderer.h"

class NoiseAwareMetropolisRenderer
  : public MultiStageMetropolisRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef MultiStageMetropolisRenderer Parent;

    /*! Null constructor does nothing.
     */
    NoiseAwareMetropolisRenderer(void);

    /*! Constructor accepts a pointer to a Scene, Film, and PathMutator.
     *  \param s Sets Parent::mRandomSequence.
     *  \param m Sets mMutator.
     *  \param importance Sets mImportance.
     *  \param target Sets mRayTarget.
     *  \param varianceExponent Sets mVarianceExponent.
     */
    NoiseAwareMetropolisRenderer(const boost::shared_ptr<RandomSequence> &sequence,
                                 const boost::shared_ptr<PathMutator> &m,
                                 const boost::shared_ptr<ScalarImportance> &importance,
                                 const unsigned int target,
                                 const float varianceExponent = 0.5f);

  protected:
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

    virtual void prepareTargetImportance(const RandomAccessFilm &mean,
                                         const RandomAccessFilm &variance,
                                         RandomAccessFilm &target) const;

    /*! The exponent to apply to the variance estimate while massaging.
     *  This defaults to 0.5 (the square root).
     */
    float mVarianceExponent;
}; // end NoiseAwareMetropolisRenderer

#endif // NOISE_AWARE_METROPOLIS_RENDERER_H

