/*! \file MipMappedRenderFilm.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a RenderFilm class which keeps its own mipmap.
 */

#ifndef MIP_MAPPED_RENDER_FILM_H
#define MIP_MAPPED_RENDER_FILM_H

#include "RenderFilm.h"

class MipMappedRenderFilm
  : public RenderFilm
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef RenderFilm Parent;

    /*! Null constructor calls the Parent.
     */
    inline MipMappedRenderFilm(void);

    /*! Constructor calls the parent and resets the statistics.
     *  \param width The new width of this RenderFilm.
     *  \param height The new height of this RenderFilm.
     *  \param filename A filename to write to post-rendering.
     */
    inline MipMappedRenderFilm(const unsigned int width,
                               const unsigned int height,
                               const std::string &filename = "");

    /*! This method deposits a Spectrum into this MipMappedRenderFilm.
     *  \param px The x-coordinate of the pixel in the unit square.
     *  \param py The y-coordinate of the pixel in the unit square.
     *  \param s The Spectrum to deposit.
     */
    inline virtual void deposit(const float px, const float py,
                                const Spectrum &s);

    /*! This method resizes this MipMappedRenderFilm.
     *  \param width The width of this MipMappedRenderFilm in pixels.
     *  \param height The height of this MipMappedRenderFilm in pixels.
     */
    inline virtual void resize(const unsigned int width,
                               const unsigned int height);

    /*! This method fills this MipMappedRenderFilm.
     *  \param v The fill value.
     */
    inline virtual void fill(const Pixel &v);

    /*! This method scales this MipMappedRenderFilm.
     *  \param s The scale value.
     */
    inline virtual void scale(const Pixel &s);

    /*! This method is called after rendering.
     */
    inline virtual void postprocess(void);

    /*! This method returns a reference to a mip level.
     *  \param i The mip map level of interest.
     *  \return A reference to mip level i.
     *  \note Mip level 0 is the finest level, and a
     *        reference to this is returned in this case.
     */
    inline RenderFilm &getMipLevel(const unsigned int i);

    /*! This method returns a const reference to a mip level.
     *  \param i The mip map level of interest.
     *  \return A reference to mip level i.
     *  \note Mip level 0 is the finest level, and a
     *        reference to this is returned in this case.
     */
    inline const RenderFilm &getMipLevel(const unsigned int i) const;

    /*! This method returns the number of mip map levels
     *  (counting the finest resolution as a level) of this
     *  MipMappedRenderFilm
     *  \return mMipMap.size() + 1
     */
    inline unsigned int getNumMipLevels(void) const;

  protected:
    // each entry of this vector is a scale factor and image
    std::vector<std::pair<float, RenderFilm> > mMipMap;
}; // end MipMappedRenderFilm

#include "MipMappedRenderFilm.inl"

#endif // MIP_MAPPED_RENDER_FILM_H

