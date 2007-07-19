/*! \file Surface.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class representing
 *         a 2D surface embedded in 3D.
 */

#ifndef SURFACE_H
#define SURFACE_H

class BoundingBox;
class Point;
class Normal;
class Ray;
class DifferentialGeometry;

class Surface
{
  public:
    /*! Null destructor does nothing.
     */
    virtual ~Surface(void);

    /*! This method returns a BoundingBox bounding this Surface.
     *  \param b The BoundingBox bounding this Surface is returned here.
     */
    virtual void getBoundingBox(BoundingBox &b) const = 0;

    /*! This method computes the intersection between the given Ray and this Surface.
     *  If an intersection exists, the 'time' of intersection is returned.
     *  \param r The Ray to intersect.
     *  \param t If an intersection exists, the parametric value of the earliest
     *           non-negative intersection is returned here.
     *  \param dg If an intersection exists, a description of this Surface's differential geometry at the intersection is returned here.
     *  \return true if r intersects this Surface; false, otherwise.
     *  \note This method must be implemented in a child class.
     */
    virtual bool intersect(const Ray &r,
                           float &t,
                           DifferentialGeometry &dg) const = 0;

    /*! This method computes whether or not an intersection between the given Ray
     *  and this Mesh exists.
     *  \param r The Ray to intersect.
     *  \return true if r intersects this Mesh; false, otherwise.
     */
    virtual bool intersect(const Ray &r) const = 0;

    /*! This method samples a point from a uniform distribution over the
     *  surface area of this Surface.
     *  \param u1 A number in [0,1).
     *  \param u2 A second number in [0,1).
     *  \param u3 A third number in [0,1).
     *  \param dg The DifferentialGeometry at the sampled Point is
     *            returned here.
     *  \param pdf The value of the surface area pdf at p is returned here.
     *  \note This method must be implemented in a derived class.
     */
    virtual void sampleSurfaceArea(const float u1,
                                   const float u2,
                                   const float u3,
                                   DifferentialGeometry &dg,
                                   float &pdf) const = 0;

    /*! This method evaluates the surface area pdf at a
     *  DifferentialGeometry of interest.
     *  \param dg The DifferentialGeometry describing the Point of interest.
     *  \return The value of the surface area pdf at dg.
     */
    virtual float evaluateSurfaceAreaPdf(const DifferentialGeometry &dg) const = 0;

    /*! This method returns the surface area of this Surface.
     *  \return The surface area of this Surface.
     *  \note This method must be implemented in a derived class.
     */
    virtual float getSurfaceArea(void) const = 0;
}; // end Surface

#endif // SURFACE_H

