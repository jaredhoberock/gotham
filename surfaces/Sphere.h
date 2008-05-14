/*! \file Sphere.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a surface abstracting
 *         a spherical object.
 */

#ifndef SPHERE_H
#define SPHERE_H

#include "Surface.h"
#include "../include/Point.h"

class Sphere
  : public Surface
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef Surface Parent;

    /*! Null constructor calls the parent.
     */
    inline Sphere(void);

    /*! Constructor accepts a center of the Sphere, and a radius.
     *  \param c The center of the Spehre, in world coordinates.
     *  \param r Sets mRadius.
     */
    Sphere(const Point &c, const float r);

    /*! This method returns a BoundingBox bounding this Sphere.
     *  \param b A BoundingBox bounding this Sphere is returned here.
     */
    virtual void getBoundingBox(BoundingBox &b) const;

    /*! This method computes the intersection between the given Ray and this Sphere.
     *  If an intersection exists, the 'time' of intersection is returned.
     *  \param r The Ray to intersect.
     *  \param t If an intersection exists, the parametric value of the earliest
     *           non-negative intersection is returned here.
     *  \param dg If an intersection exists, a description of this Sphere's differential geometry at the intersection is returned here.
     *  \return true if r intersects this Sphere; false, otherwise.
     *  \note This method must be implemented in a child class.
     */
    virtual bool intersect(const Ray &r,
                           float &t,
                           DifferentialGeometry &dg) const;

    /*! This method computes whether or not an intersection between the given Ray
     *  and this Sphere exists.
     *  \param r The Ray to intersect.
     *  \return true if r intersects this Sphere; false, otherwise.
     */
    virtual bool intersect(const Ray &r) const;

    /*! This method samples a Point from a uniform distribution over the
     *  surface area of this Sphere.
     *  \param u1 A number on [0,1).
     *  \param u2 A second number in [0,1).
     *  \param u3 A third number in [0,1).
     *  \param dg The DifferentialGeometry at the sampled Point is
     *            returned here.
     *  \param pdf The value of the surface area measure pdf at dg is
     *             returned here.
     */
    virtual void sampleSurfaceArea(const float u1,
                                   const float u2,
                                   const float u3,
                                   DifferentialGeometry &dg,
                                   float &pdf) const;

    /*! This method evaluates the surface area pdf at a DifferentialGeometry
     *  of interest.
     *  \param dg The DifferentialGeometry describing the Point of interest.
     *  \return The value of the surface area pdf at dg.
     */
    virtual float evaluateSurfaceAreaPdf(const DifferentialGeometry &dg) const;

    /*! This method returns the surface area of this Sphere.
     *  \return mSurfaceArea.
     */
    virtual float getSurfaceArea(void) const;

    /*! This method returns the reciprocal of surface area of this Sphere.
     *  \return mOneOverSurfaceArea
     */
    virtual float getInverseSurfaceArea(void) const;

    /*! This method gets the DifferentialGeometry at the given
     *  Point and ParametricCoordinates of interest.
     *  \param p The Point of interest on this Sphere.
     *  \param dg The DifferentialGeometry at p is returned here.
     *  \note This method assumes p lies on the surface area of
     *        this Sphere.
     */
    virtual void getDifferentialGeometry(const Point &p,
                                         DifferentialGeometry &dg) const;

  protected:
    /*! A Sphere has a radius.
     */
    float mRadius;

    /*! A Sphere has a center.
     */
    Point mCenter;

    /*! A Sphere has a surface area.
     */
    float mSurfaceArea;

    /*! The reciprocal of mSurfaceArea.
     */
    float mInverseSurfaceArea;
}; // end Sphere

#endif // SPHERE_H

