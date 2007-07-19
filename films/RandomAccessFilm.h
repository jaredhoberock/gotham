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

    /*! This method returns a reference to the pixel
     *  at raster location (px,py).
     *  \param px A raster coordinate in [0,mWidth).
     *  \param py A raster coordinate in [0,mHeight).
     *  \return A reference to the pixel at (px,py).
     */
    inline Pixel &raster(const unsigned int px,
                         const unsigned int py);

    /*! This method returns a const reference to the pixel
     *  at raster location (px,py).
     *  \param px A raster coordinate in [0,mWidth).
     *  \param py A raster coordinate in [0,mHeight).
     *  \return A const reference to the pixel at (px,py).
     */
    inline const Pixel &raster(const unsigned int px,
                               const unsigned int py) const;

    /*! This method fills this Film with the given
     *  pixel value.
     *  \param v The fill value.
     */
    inline void fill(const Pixel &v);

  private:
    typedef Array2<Pixel> Parent1;
}; // end RandomAccessFilm

#include "RandomAccessFilm.inl"

#endif // RANDOM_ACCESS_FILM_H

