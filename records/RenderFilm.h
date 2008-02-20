/*! \file RenderFilm.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a random access film
 *         which keeps statistics useful to a rendering
 *         process.
 */

#ifndef RENDER_FILM_H
#define RENDER_FILM_H

#include "Record.h"
#include "RandomAccessFilm.h"

class RenderFilm
  : public Record,
    public RandomAccessFilm
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef Record Parent0;

    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef RandomAccessFilm Parent1;

    /*! Null constructor calls the parent.
     */
    inline RenderFilm(void);

    /*! Constructor calls the parent and resets the statistics.
     *  \param width The new width of this RenderFilm.
     *  \param height The new height of this RenderFilm.
     *  \param filename A filename to write to post-rendering.
     */
    inline RenderFilm(const unsigned int width,
                      const unsigned int height,
                      const std::string &filename = "");

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

    /*! This method records the square of the given Path's results
     *  by repeatedly calling deposit() for each Result in results.
     *  \param w The weight to associate with the record.
     *  \param x The HyperPoint associated with xPath.
     *  \param xPath The Path to record.
     *  \param results A list of PathSampler::Results to record.
     */
    inline virtual void recordSquare(const float w,
                                     const PathSampler::HyperPoint &x,
                                     const Path &xPath,
                                     const std::vector<PathSampler::Result> &results);

    /*! This method interprets this RenderFilm as an estimate of variance per pixel
     *  and updates the variance of the given deposit.
     *  \param w The weight to associate with the record.
     *  \param x The HyperPoint associated with xPath.
     *  \param xPath The path to record.
     *  \param meanImage The mean image (the image currently being rendered).
     *  \param i The index of this sample.
     */
    inline virtual void recordVariance(const float w,
                                       const PathSampler::HyperPoint &x,
                                       const Path &xPath,
                                       const std::vector<PathSampler::Result> &results,
                                       const RenderFilm &meanImage,
                                       const size_t i);

    /*! This method calls Parent::resize() and resets the
     *  statistics.
     *  \param width The new width of this RenderFilm.
     *  \param height The new height of this RenderFilm.
     */
    inline virtual void resize(const unsigned int width,
                               const unsigned int height);

    /*! This method deposits a Spectrum into this RenderFilm.
     *  \param px The x-coordinate of the pixel in the unit square.
     *  \param py The y-coordinate of the pixel in the unit square.
     *  \param s The Spectrum to deposit.
     */
    inline virtual void deposit(const float px, const float py,
                                const Spectrum &s);

    /*! This method deposits a Spectrum into this RenderFilm at a raster location.
     *  \param rx The x-index of the raster location in the grid.
     *  \param ry The y-index of the raster location in the grid.
     *  \param s The Spectrum to deposit.
     */
    inline virtual void deposit(const size_t rx, const size_t ry,
                                const Spectrum &s);

    /*! This method returns the sum of all deposits into this RenderFilm since
     *  the last resize() event.
     *  \return mSum.
     */
    inline const Spectrum &getSum(void) const;

    /*! This method returns the sum of the logarithm of the Y channel of all deposits
     *  into this RenderFilm since the last resize() event.
     *  \returm mSumLogLuminance.
     */
    inline float getSumLogLuminance(void) const;

    /*! This method returns the number of deposits into this RenderFilm since
     *  the last resize() event.
     *  \return mNumDeposits.
     */
    inline size_t getNumDeposits(void) const;

    /*! This method calls the Parent and sets the statistics accordingly.
     *  \param v The fill value.
     */
    inline virtual void fill(const Pixel &v);

    /*! This method calls the Parent and sets the statistics accordingly.
     *  \param s The scale value.
     */
    inline virtual void scale(const float s);

    /*! This method calls the Parent and sets the statistics accordingly.
     *  \param s The scale value.
     */
    inline virtual void scale(const Pixel &s);

    /*! This method returns the maximum luminance over pixels.
     *  \return mMaximumLuminance.
     */
    inline float getMaximumLuminance(void) const;

    /*! This method returns the minimum luminance over pixels.
     *  \return mMinimumLuminance.
     */
    inline float getMinimumLuminance(void) const;

    /*! Grant access to const pixel().
     *  This method returns a reference to the pixel
     *  at location (u,v)
     *  \param u A parameteric coordinate in [0,1).
     *  \param v A parameteric coordinate in [0,1).
     *  \return A reference to the pixel at (u,v)
     */
    inline const Parent1::Pixel &pixel(const float u,
                                       const float v) const;

    /*! Grant access to const raster().
     *  This method returns a reference to the pixel
     *  at raster location (i,j)
     *  \param i The column index of the raster location of interest.
     *  \param j The row index of the raster location of interest.
     *  \return A const reference to the pixel at raster location (i,j).
     */
    inline const Parent1::Pixel &raster(const size_t i,
                                        const size_t j) const;

    /*! This method is called before rendering and clears
     *  this RenderFilm to black.
     */
    inline virtual void preprocess(void);

    /*! This method is called after rendering.
     */
    inline virtual void postprocess(void);
    
    // XXX DESIGN this is kludgy
    //     i'm adding this so that GpuFilm has something
    //     to override later
    inline virtual void init(void);

    /*! This method returns the filename.
     *  \return mFilename
     */
    inline const std::string &getFilename(void) const;

    /*! This method sets mFilename.
     *  \param filename Sets mFilename.
     */
    inline void setFilename(const std::string &filename);

    /*! This method sets mApplyTonemap.
     *  \param a Sets mApplyTonemap.
     */
    inline void setApplyTonemap(const bool a);

  protected:
    /*! Hide access to non-const pixel().
     */
    using Parent1::pixel;

    /*! XXX TODO: hide access to raster().
     */

    /*! This method resets the statistics.
     */
    inline void reset(void);

    /*! The number of times deposit() has been called
     *  since the last resize() event.
     */
    size_t mNumDeposits;

    /*! The sum of all calls to deposit().
     */
    Spectrum mSum;

    /*! The sum of log luminance of all calls to deposit().
     */
    float mSumLogLuminance;

    /*! The maximum luminance over pixels.
     */
    float mMaximumLuminance;

    /*! The minimum luminance over pixels.
     */
    float mMinimumLuminance;

    /*! A filename to write to during postprocessing.
     */
    std::string mFilename;

    /*! Whether or not to apply the Reinhard tonemap on postprocess.
     */
    bool mApplyTonemap;
}; // end RenderFilm

#include "RenderFilm.inl"

#endif // RENDER_FILM_H

