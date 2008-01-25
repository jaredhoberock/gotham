/*! \file NoiseAwareMetropolisRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a MultiStageMetropolisRenderer
 *         which targets perceptually significant image noise.
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
     *  \param varianceExponent Sets mVarianceExponent.
     */
    NoiseAwareMetropolisRenderer(const boost::shared_ptr<RandomSequence> &sequence,
                                 const boost::shared_ptr<PathMutator> &m,
                                 const boost::shared_ptr<ScalarImportance> &importance,
                                 const float varianceExponent = 0.5f);

    /*! This method sets the filename to write to for the target density.
     *  \param filename The name of the file to create and write the final target
     *         sampling density to.
     */
    void setTargetFilename(const std::string &filename);

  protected:
    /*! This method updates this NoiseAwareMetropolisRenderer's importance
     *  function.
     *  \param The estimate of b for the luminance importance function.
     *  \param w The width, in pixels, of the importance function to create.
     *  \param h The height, in pixels, of the importance function to create.
     *  \param x The current state.
     *  \param xPath The current Path.
     *  \param xResults xPath's results.
     *  \param ix The new importance of x is returned here.
     *  \param xPdf The new pdf of x is returned here.
     *  \return The inverse of the new importance function's normalization constant
     *          is returned here.
     */
    float updateImportance(const float bLuminance,
                           const float w,
                           const float h,
                           const PathSampler::HyperPoint &x,
                           const Path &xPath,
                           const std::vector<PathSampler::Result> &xResults,
                           float &ix,
                           float &xPdf);

    /*! This method prepares a desired sampling density to target.
     *  \param mean The current mean estimate.
     *  \param varianceExponent The current variance estimate.
     *  \param target The desired sampling density is returned here.
     */
    virtual void prepareTargetImportance(const RandomAccessFilm &mean,
                                         const RandomAccessFilm &variance,
                                         RandomAccessFilm &target) const;

    float mVarianceExponent;

    std::string mTargetFilename;
}; // end NoiseAwareMetropolisRenderer

#endif // NOISE_AWARE_METROPOLIS_RENDERER_H

