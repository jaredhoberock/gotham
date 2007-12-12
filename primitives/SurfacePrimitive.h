/*! \file SurfacePrimitive.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a Primitive which is geometrically
 *         described by a Surface.
 */

#ifndef SURFACE_PRIMITIVE_H
#define SURFACE_PRIMITIVE_H

#include "Primitive.h"
#include "../shading/Material.h"
#include <boost/shared_ptr.hpp>
#include "../surfaces/Surface.h"

class SurfacePrimitive
  : public Primitive
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Primitive Parent;

    /*! Constructor accepts a Surface and a Material.
     *  \param s Sets mSurface.
     *  \param m Sets mMaterial.
     */
    SurfacePrimitive(boost::shared_ptr<Surface> s,
                     boost::shared_ptr<Material> m);

    /*! Null destructor does nothing.
     */
    virtual ~SurfacePrimitive(void);

    /*! This method returns a const pointer to mMaterial.
     *  \return mMaterial.
     *  XXX Should this return a shared_ptr?
     */
    const Material *getMaterial(void) const;

    /*! This method returns a const pointer to mSurface.
     *  \return mSurface.
     */
    const Surface *getSurface(void) const;

    /*! This method returns a shared pointer to mSurface.
     *  \param s mSurface is returned here.
     */
    void getSurface(boost::shared_ptr<Surface> &s) const;

    /*! This method returns the world space bounding box of mSurface.
     *  \param b Set to mShape->getBoundingBox().
     */
    void getBoundingBox(BoundingBox &b) const;

    /*! This method computes the intersection between a Ray and this
     *  SurfacePrimitive's Surface.
     *  \param r The Ray to intersect.  If an intersection exists, r's legal parametric interval is altered accordingly.
     *  \param inter If an Intersection exists, information regarding the first intersection along r is returned here.
     *  \return true if an Intersection exists; false, otherwise.
     */
    virtual bool intersect(Ray &r, Intersection &inter) const;

    /*! This method computes whether or not an intersection between the given
     *  Ray and mSurface exists.
     *  \return r The Ray to intersect.
     *  \return true if an intersection between r and mSurface exists; false, otherwise.
     */
    virtual bool intersect(const Ray &r) const;

    /*! This method returns the surface area of this SurfacePrimitive's
     *  Surface.
     *  \return mSurface->getSurfaceArea()
     *  XXX Should we include this?
     */
    float getSurfaceArea(void) const;

    /*! This method returns the inverse surface area of this SurfacePrimitive's
     *  Surface.
     *  \return mSurface->getInverseSurfaceArea()
     *  XXX Should we include this?
     */
    float getInverseSurfaceArea(void) const;

    /*! This method samples this SurfacePrimitive's surface area.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param u2 A third real number in [0,1).
     *  \param dg The DifferentialGeometry at the sampled Point is
     *            returned here.
     *  \param pdf
     *  XXX should we include this?
     */
    void sampleSurfaceArea(const float u0,
                           const float u1,
                           const float u2,
                           DifferentialGeometry &dg,
                           float &pdf) const;

    /*! This method evaluates the surface area pdf of choosing the
     *  given point on this SurfacePrimitive.
     *  \param dg The DifferentialGeometry at the point of interest.
     *  \return The surface area measure pdf of choosing dg on this
     *          SurfacePrimitive.
     */
    virtual float evaluateSurfaceAreaPdf(const DifferentialGeometry &dg) const;

  protected:
    /*! A SurfacePrimitive keeps a pointer to a Surface.
     */
    boost::shared_ptr<Surface> mSurface;

    /*! A SurfacePrimitive keeps a pointer to a Material.
     */
    boost::shared_ptr<Material> mMaterial;
}; // end SurfacePrimitive

#endif // SURFACE_PRIMITIVE_H

