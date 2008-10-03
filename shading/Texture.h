/*! \file Texture.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class abstracting
 *         textures for shading.
 */

#pragma once

#include <array2/Array2.h>
#include "../include/detail/Spectrum.h"

class Texture
  : protected Array2<Spectrum>
{
  public:
    /*! Null constructor creates a 1x1 white Texture.
     */
    Texture(void);

    /*! Constructor calls the Parent.
     *  \param w The width of the Texture.
     *  \param h The height of the Texture.
     *  \param pixels The pixel data to copy into this Texture.
     */
    Texture(const size_t w, const size_t h, const Spectrum *pixels);

    /*! Constructor takes a filename referring to
     *  an image file on disk.
     *  \param filename The name of the image file of interest.
     */
    Texture(const char *filename);

    /*! This method provides const access to pixels.
     *  \param x The column index of the pixel of interest.
     *  \param y The row index of the pixel of interest.
     *  \return A const reference to pixel (x,y).
     */
    virtual const Spectrum &texRect(const size_t x,
                                    const size_t y) const;

    /*! This method provides nearest-neighbor filtering
     *  given pixel coordinates in [0,1]^2.
     *  \param u The u-coordinate of the pixel location of interest.
     *  \param v The v-coordinate of the pixel location of interest.
     *  \return The box-filtered pixel at (u,v).
     */
    virtual const Spectrum &tex2D(const float u,
                                  const float v) const;

    /*! This method loads this Texture's data from an
     *  image file on disk.
     *  \param filename 
     *  \param filename The name of the image file of interest.
     */
    virtual void load(const char *filename);

  protected:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Array2<Spectrum> Parent;
}; // end Texture

