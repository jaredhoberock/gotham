/*! \file RasterizableSurfacePrimitive.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a SurfacePrimitive class
 *         which implements Rasterizable.
 */

#ifndef RASTERIZABLE_SURFACE_PRIMITIVE_H
#define RASTERIZABLE_SURFACE_PRIMITIVE_H

#include "../primitives/SurfacePrimitive.h"
#include "Rasterizable.h"

class RasterizableSurfacePrimitive
  : public SurfacePrimitive,
    public Rasterizable
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef SurfacePrimitive Parent0;

    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef Rasterizable Parent1;

    /*! Constructor accepts a Surface and a Material.
     *  \param s Sets Parent0::mSurface.
     *  \param m Sets Parent0::mMaterial.
     */
    RasterizableSurfacePrimitive(boost::shared_ptr<Surface> s,
                                 const MaterialHandle &m);

    /*! This method calls Parent0::mSurface->rasterize() if
     *  Parent0::mSurface implements Rasterizable.
     */
    virtual void rasterize(void);
}; // end RasterizableSurfacePrimitive

#endif // RASTERIZABLE_SURFACE_PRIMITIVE_H

