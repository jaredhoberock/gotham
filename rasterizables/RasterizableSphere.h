/*! \file RasterizableSphere.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Sphere
 *         class which implements Rasterizable.
 */

#ifndef RASTERIZABLE_SPHERE_H
#define RASTERIZABLE_SPHERE_H

#include "Rasterizable.h"
#include "../surfaces/Sphere.h"

class RasterizableSphere
  : public Sphere,
    public Rasterizable
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef Sphere Parent0;

    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef Rasterizable Parent1;

    /*! Constructor accepts a center of the RasterizableSphere, and a radius.
     *  \param c The center of this RasterizableSphere.
     *  \param r The radius of this RasterizableSphere.
     */
    RasterizableSphere(const Point &c,
                       const float r);

    /*! This method rasterizes this RasterizableSphere using
     *  OpenGL commands.
     */
    virtual void rasterize(void);
}; // end RasterizableSphere

#endif // RASTERIZABLE_SPHERE_H

