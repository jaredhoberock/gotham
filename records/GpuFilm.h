/*! \file GpuFilm.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Film class whose
 *         data resides on the GPU.
 */

#ifndef GPU_FILM_H
#define GPU_FILM_H

#include <gl++/texture/Texture.h>
#include <gl++/framebuffer/Framebuffer.h>
#include "Film.h"
#include "../include/Spectrum.h"
#include <vector>
#include <boost/thread/mutex.hpp>

template<typename ParentFilmType>
  class GpuFilm
    : public ParentFilmType
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ParentFilmType Parent;

    /*! Null constructor calls the Parent.
     */
    inline GpuFilm(void);

    /*! This method resizes this GpuFilm.
     *  \param width The width of this GpuFilm in pixels.
     *  \param height The height of this GpuFilm in pixels.
     */
    inline virtual void resize(const unsigned int width,
                               const unsigned int height);
    
    /*! This method initializes this GpuFilm by creating
     *  an OpenGL identifier for mTexture and allocating
     *  storage.
     *  \note This method assumes an OpenGL context exists.
     */
    inline virtual void init(void);

    /*! This method deposits a Spectrum into this RenderFilm.
     *  \param px The x-coordinate of the pixel in the unit square.
     *  \param py The y-coordinate of the pixel in the unit square.
     *  \param s The Spectrum to deposit.
     */
    using Parent::deposit;
    inline virtual void deposit(const float px, const float py,
                                const Spectrum &s);

    /*! This method fills this Film with the given
     *  pixel value.
     *  \param v The fill value.
     */
    inline void fill(const typename Parent::Pixel &v);

    /*! This method scales this Film by the given Pixel value.
     *  \param s The scale value.
     */
    inline void scale(const typename Parent::Pixel &s);

  //protected:
    /*! This method renders all pending deposits to mTexture.
     */
    virtual void renderPendingDeposits(void);

    /*! Pixel storage.
     */
    glpp::Texture mTexture;

    /*! A framebuffer object.
     */
    Framebuffer mFramebuffer;

    /*! \typedef Deposit
     *  \brief Shorthand.
     */
    typedef std::pair<gpcpu::float2, Spectrum> Deposit;

    /*! A buffer of deposits.
     */
    std::vector<Deposit> mDepositBuffer;

    // XXX this might be too heavy weight for frequent things
    //     such as deposits
    boost::mutex mMutex;
}; // end GpuFilm

#include "GpuFilm.inl"

#endif // GPU_FILM_H

