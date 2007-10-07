/*! \file RenderFilm.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a random access film
 *         which keeps statistics useful to a rendering
 *         process.
 */

#ifndef RENDER_FILM_H
#define RENDER_FILM_H

#include "RandomAccessFilm.h"

class RenderFilm
  : public RandomAccessFilm
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef RandomAccessFilm Parent;

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

    /*! This method returns the number of deposits into this RenderFilm since
     *  the last resize() event.
     *  \return mNumDeposits.
     */
    inline size_t getNumDeposits(void) const;

    /*! This method calls the Parent and sets the statistics accordingly.
     *  \param v The fill value.
     */
    inline void fill(const Pixel &v);

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
    inline const Parent::Pixel &pixel(const float u,
                                      const float v) const;

    /*! This method is called after rendering.
     */
    inline virtual void postprocess(void);

    /*! This method returns the filename.
     *  \return mFilename
     */
    inline const std::string &getFilename(void) const;

    /*! This method writes this RenderFilm to the given filename.
     *  \param filename The name of the file to write to.
     *  XXX DESIGN I still don't think this is the place for this.
     */
    inline void writeEXR(const char *filename) const;

  protected:
    /*! Hide access to non-const pixel().
     */
    using Parent::pixel;

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

    /*! The maximum luminance over pixels.
     */
    float mMaximumLuminance;

    /*! The minimum luminance over pixels.
     */
    float mMinimumLuminance;

    /*! A filename to write to during postprocessing.
     */
    std::string mFilename;
}; // end RenderFilm

#include "RenderFilm.inl"

#endif // RENDER_FILM_H

