/*! \file CudaGotham.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a CUDA
 *         version of the Gotham API.
 */

#pragma once

#include "../../api/Gotham.h"

class CudaGotham
  : public Gotham
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Gotham Parent;

    /*! Null constructor calls Parent().
     */
    CudaGotham(void);

    /*! Copy constructor copies from a Parent type.
     *  \param g The Parent to copy from.
     */
    CudaGotham(const Parent &g);

    /*! This method starts a render.
     */
    virtual void render(void);

    /*! This method creates a new Sphere.
     *  \param cx The x-coordinate of the center of the Sphere.
     *  \param cy The y-coordinate of the center of the Sphere.
     *  \param cz The z-coordinate of the center of the Sphere.
     *  \param radius The radius of the Sphere.
     */
    virtual void sphere(const float cx,
                        const float cy,
                        const float cz,
                        const float radius);
}; // end CudaGotham

