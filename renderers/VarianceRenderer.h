/*! \file VarianceRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Monte Carlo path tracer
 *         which also estimates variance per pixel.
 */

#ifndef VARIANCE_RENDERER_H
#define VARIANCE_RENDERER_H

#include "PathDebugRenderer.h"
#include <array2/Array2.h>

class VarianceRenderer
  : public PathDebugRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef PathDebugRenderer Parent;

    /*! Null constructor does nothing.
     */
    VarianceRenderer(void);

    /*! Constructor accepts a RandomSequence and a PathSampler
     *  and calls the null constructor of the Parent.
     *  \param s Sets Parent::mRandomSequence.
     *  \param sampler Sets mSampler.
     */
    VarianceRenderer(const boost::shared_ptr<RandomSequence> &s,
                     const boost::shared_ptr<PathSampler> &sampler);

    /*! This method sets this VarianceRenderer's variance Record.
     *  \param r Sets mVarianceRecord
     */
    void setVarianceRecord(const boost::shared_ptr<Record> &r);

  protected:
    /*! This method calls mVarianceRecord->preprocess()
     *  and then calls the Parent.
     */
    virtual void preprocess(void);

    /*! This method calls the Parent and then calls
     *  mVarianceRecord->postprocess().
     */
    virtual void postprocess(void);

    /*! This method performs the main rendering work.
     *  \param progress A ProgressCallback which is periodically
     *         updated during the render.
     */
    virtual void kernel(ProgressCallback &progress);

    /*! This method computes and records a pixel's variance
     *  \param px The x-position of the pixel's raster location.
     *  \param py The y-position of the pixel's raster location.
     *  \param n The number of samples generated.
     *  \param samples The non-zero sample values.
     */
    virtual void depositVariance(const unsigned int px,
                                 const unsigned int py,
                                 const unsigned int n,
                                 const std::vector<Spectrum> &samples);

    virtual void depositVariance(const PathSampler::HyperPoint &x,
                                 const Path &xPath,
                                 const std::vector<PathSampler::Result> &results,
                                 Spectrum oldMean);


    /*! Variance is recorded here.
     */
    boost::shared_ptr<Record> mVarianceRecord;

    /*! This array logs samples per pixel.
     */
    Array2<float> mSamplesImage;
}; // end VarianceRenderer

#endif // VARIANCE_RENDERER_H

