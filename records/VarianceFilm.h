/*! \file VarianceFilm.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a RenderFilm
 *         which can simultaneously approximate a
 *         consistent? estimate of per-pixel variance
 */

#ifndef VARIANCE_FILM_H
#define VARIANCE_FILM_H

#include "RenderFilm.h"

class VarianceFilm
  : public RenderFilm
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef RenderFilm Parent;

    /*! Null constructor calls the parent.
     */
    inline VarianceFilm(void);

    /*! Constructor calls the parent and resets the statistics.
     *  \param width The new width of this RenderFilm.
     *  \param height The new height of this RenderFilm.
     *  \param estimate An image containing a per-pixel estimate of the
     *                  image to render.
     *  \param filename A filename to write to post-rendering.
     *  \param varianceFilename A filename to write the variance image to
     *                          post-rendering.
     */
    inline VarianceFilm(const unsigned int width,
                        const unsigned int height,
                        const boost::shared_ptr<RandomAccessFilm> &estimate,
                        const std::string &filename = "",
                        const std::string &varianceFilename = "");

    /*! This method records the given Path by repeatedly calling
     *  deposit() for each Result in results.
     *  \param w The weight to associate with the record.
     *  \param x The HyperPoint associated with xPath.
     *  \param xPath The Path to record.
     *  \param results A list of PathSampler::Results to record.
     */
    inline virtual void record(const float w,
                               const PathSampler::HyperPoint &x,
                               const Path &xPath,
                               const std::vector<PathSampler::Result> &results);

    /*! This method calls Parent::resize() and mVariance.resize().
     *  \param width The new width of this RenderFilm.
     *  \param height The new height of this RenderFilm.
     */
    inline virtual void resize(const unsigned int width,
                               const unsigned int height);

    /*! This method calls the Parent and preprocesses mVariance.
     */
    inline virtual void preprocess(void);

    /*! This method calls the Parent and postprocesses mVariance.
     */
    inline virtual void postprocess(void);

    /*! This method calls the Parent and mVariance.fill()
     *  \param v The fill value.
     */
    inline virtual void fill(const Pixel &v);

    /*! This method calls the Parent and mVariance.scale()
     *  \param s The scale value.
     */
    inline virtual void scale(const Pixel &s);

  protected:
    /*! An estimate of the mean.
     */
    boost::shared_ptr<RandomAccessFilm> mMeanEstimate;

    /*! The image containing the variance image.
     */
    RenderFilm mVariance;
}; // end VarianceFilm

#include "VarianceFilm.inl"

#endif // VARIANCE_FILM_H

