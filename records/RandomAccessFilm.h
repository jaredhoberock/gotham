/*! \file RandomAccessFilm.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class abstracting the
 *         Film of a Camera.
 */

#ifndef RANDOM_ACCESS_FILM_H
#define RANDOM_ACCESS_FILM_H

#include <vector>
#include <spectrum/Spectrum.h>
#include "Film.h"
#include <array2/Array2.h>

class RandomAccessFilm
  : public Film,
    private Array2<Spectrum>
{
  public:
    typedef Film Parent0;

    typedef Spectrum Pixel;

    /*! Null constructor calls the Parents.
     */
    inline RandomAccessFilm(void);

    /*! Constructor accepts a width and height for
     *  this Film.
     *  \param width The width of this RandomAccessFilm in pixels.
     *  \param height The height of this Film in pixels.
     */
    inline RandomAccessFilm(const unsigned int width,
                            const unsigned int height);

    /*! This method resizes this Film.
     *  \param width The width of this Film in pixels.
     *  \param height The height of this Film in pixels.
     */
    inline virtual void resize(const unsigned int width,
                               const unsigned int height);

    /*! This method returns a reference to the pixel
     *  at location (u,v)
     *  \param u A parameteric coordinate in [0,1).
     *  \param v A parameteric coordinate in [0,1).
     *  \return A reference to the pixel at (u,v)
     */
    inline Pixel &pixel(const float u,
                        const float v);

    /*! This method returns a const reference to the pixel
     *  at location (u,v)
     *  \param u A parameteric coordinate in [0,1).
     *  \param v A parameteric coordinate in [0,1).
     *  \return A reference to the pixel at (u,v)
     */
    inline const Pixel &pixel(const float u,
                              const float v) const;

    /*! This method returns a bilinearly interpolated value
     *  from the pixel location (u,v).
     *  \param u A parametric coordinate in [0,1).
     *  \param v A parametric coordinate in [0,1).
     *  \return A bilinearly interpolated value constructed from
     *          the neighborhood of (u,v).
     */
    inline Pixel bilerp(const float u, const float v) const;

    /*! This method returns a reference to the pixel
     *  at raster location (px,py).
     *  \param px A raster coordinate in [0,mWidth).
     *  \param py A raster coordinate in [0,mHeight).
     *  \return A reference to the pixel at (px,py).
     */
    inline Pixel &raster(const size_t px,
                         const size_t py);

    /*! This method returns a const reference to the pixel
     *  at raster location (px,py).
     *  \param px A raster coordinate in [0,mWidth).
     *  \param py A raster coordinate in [0,mHeight).
     *  \return A const reference to the pixel at (px,py).
     */
    inline const Pixel &raster(const size_t px,
                               const size_t py) const;

    /*! This method fills this Film with the given
     *  pixel value.
     *  \param v The fill value.
     */
    inline void fill(const Pixel &v);

    /*! This method scales this Film's pixels by the given value.
     *  \param s The scale value.
     */
    inline void scale(const Pixel &s);

    /*! This method computes the sum of each Pixel value.
     *  \return The sum over all Pixels.
     */
    inline Pixel computeSum(void) const;

    /*! This method computes the mean Pixel value.
     *  \return computeSum() / (getWidth() * getHeight())
     */
    inline Pixel computeMean(void) const;

    /*! This method applies the tonemap operator of Reinhard et al 2002
     *  to this RandomAccessFilm.
     *  XXX DESIGN this probably doesn't belong here.
     */
    inline void tonemap(void);

    /*! This method subtracts a RandomAccessFilm pixel by pixel
     *  from this RandomAccessFilm.
     *  \param rhs The RandomAccessFilm to subtract.
     *  \return *this
     *  \note If rhs is not the same dimensions as this RandomAccessFilm,
     *        this method does nothing.
     */
    inline RandomAccessFilm &operator-=(const RandomAccessFilm &rhs);

    /*! This method maps a coordinate in the unit square to a raster position
     *  of this RandomAccessFilm.
     *  \param u The first coordiante in the unit square.
     *  \param v The second coordinate in the unit square.
     *  \param i The raster column corresponding to u is returned here.
     *  \param j The raster column corresponding to v is returned here.
     */
    inline void getRasterPosition(const float u, const float v,
                                  size_t &i, size_t &j) const;

    /*! This method writes this RandomAccessFilm to the given filename.
     *  \param filename The name of the file to write to.
     *  XXX DESIGN why should this be a member
     */
    inline void writeEXR(const char *filename) const;

    /*! This method reads this RandomAccessFilm from the given filename.
     *  \param filename The name of the file to read from.
     *  XXX DESIGN why should this be a member
     */
    inline void readEXR(const char *filename);

    /*! This method resamples this RandomAccessFilm into the given target.
     *  \param target The RandomAccessFilm to resample to.
     */
    inline void resample(RandomAccessFilm &target) const;

    /*! This method integrates a rectangle in this RandomAccessFilm and returns
     *  the value
     *  \param xStart The x-coordinate of the lower left hand corner of the rectangle.
     *  \param yStart The y-coordinate of the lower left hand corner of the rectangle.
     *  \param xEnd The x-coordinate of the upper right hand corner of the rectangle.
     *  \param yEnd The y-coordinate of the upper right hand corner of the rectangle.
     *  \param integral The result of integration is returned here.
     */
    inline void integrateRectangle(const float xStart, const float yStart,
                                   const float xEnd, const float yEnd,
                                   Pixel &integral) const;

    /*! This method erodes the holes of this RandomAccessFilm by setting each hole pixel
     *  to the average of its neighbors.
     *  \param h The hole value.
     *  \return The number of remaining hole pixels after erosion.
     */
    inline size_t erode(const Pixel &h);

    /*! This method performs bilateral filtering on this RandomAccessFilm.
     *  \param sigmad The standard deviation of the spacial gaussian kernel.
     *  \param sigmar The standard deviation of the intensity gaussian kernel.
     *  \param intensity An image to use for intensity similarity.
     *  \note sigmad & sigmar correspond to those quantities from
     *        Tomasi & Manduchi '98
     */
    inline void bilateralFilter(const float sigmad,
                                const float sigmar,
                                const RandomAccessFilm &intensity);

  private:
    typedef Array2<Pixel> Parent1;
}; // end RandomAccessFilm

#include "RandomAccessFilm.inl"

#endif // RANDOM_ACCESS_FILM_H

