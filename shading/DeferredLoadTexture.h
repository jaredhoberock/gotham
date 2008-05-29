/*! \file DeferredLoadTexture.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Texture class
 *         which is lazily loaded from a file upon first
 *         pixel access.
 */

#pragma once

#include "Texture.h"

class DeferredLoadTexture
  : public Texture
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Texture Parent;

    /*! Constructor takes a filename to load from.
     *  \param filename The file of interest.
     */
    DeferredLoadTexture(const char *filename);

    /*! This method provides const access to pixels.
     *  \param x The column index of the pixel of interest.
     *  \param y The row index of the pixel of interest.
     *  \return A const reference to pixel (x,y).
     */
    virtual const Spectrum &texRect(const size_t x,
                                    const size_t y) const;

  protected:
    /*! This method loads this DeferredLoadTexture's data
     *  from mFilename.
     */
    virtual void load(void);

    mutable bool mNeedsLoad;
    mutable std::string mFilename;
}; // end DeferredLoadTexture

