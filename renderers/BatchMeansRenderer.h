/*! \file BatchMeansRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a MetropolisRenderer
 *         which also estimates per-pixel variance with
 *         the batch means method.
 */

#ifndef BATCH_MEANS_RENDERER_H
#define BATCH_MEANS_RENDERER_H

#include "MetropolisRenderer.h"

class BatchMeansRenderer
  : public MetropolisRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef MetropolisRenderer Parent;

    /*! Null constructor does nothing.
     */
    BatchMeansRenderer(void);

    /*! Constructor accepts a PathMutator and calls the
     *  null constructor of the Parent.
     *  \param s Sets Parent::mRandomSequence.
     *  \param m Sets mMutator.
     *  \param i Sets mImportance.
     */
    BatchMeansRenderer(const boost::shared_ptr<RandomSequence> &s,
                       const boost::shared_ptr<PathMutator> &mutator,
                       const boost::shared_ptr<ScalarImportance> &importance);

  protected:
    /*! This method coordinates preprocessing tasks prior to rendering.
     */
    virtual void preprocess(void);

    /*! This method renders mScene to mFilm.
     *  \param progress A callback, which will be periodically
     *                  called throughout the rendering process.
     */
    virtual void kernel(ProgressCallback &progress);

    /*! This method calls the Parent and writes mAcceptanceImage.
     */
    virtual void postprocess(void);

    /*! This method updates the current variance estimate.
     *  \param samplesPerBatch The average number of samples per batch mean.
     */
    void updateVarianceEstimate(const float samplesPerBatch);

    /*! This image contains the mean over batches.
     */
    RenderFilm mMeanOverBatches;

    /*! This image contains a 'slice' of the current render.
     */
    RenderFilm mCurrentBatch;

    /*! This image contains an estimate of the variance of each pixel of
     *  the current render, consistently approximated by the batch means
     *  technique.
     */
    RenderFilm mVarianceImage;

    /*! This counts the number of batches.
     */
    unsigned int mNumBatches;
}; // end BatchMeansRenderer

#endif // BATCH_MEANS_RENDERER_H

