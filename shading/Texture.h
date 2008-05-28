/*! \file Texture.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class abstracting
 *         textures for shading.
 */

#pragma once

#include <array2/Array2.h>
#include "../include/Spectrum.h"

class Texture
  : public Array2<Spectrum>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Array2<Spectrum> Parent;

    /*! Null constructor creates a 1x1 white Texture.
     */
    Texture(void);

    /*! Constructor calls the Parent.
     *  \param w The width of the Texture.
     *  \param h The height of the Texture.
     *  \note Pixel colors are left undefined.
     */
    Texture(const size_t w, const size_t h);

    /*! Constructor takes a filename referring to
     *  an image file on disk.
     *  \param filename The name of the image file of interest.
     */
    Texture(const char *filename);
}; // end Texture

