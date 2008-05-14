/*! \file BoundingBox.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class abstracting an axis-aligned bounding box.
 */

#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include "../include/Point.h"
class Ray;
class Normal;
#include "AxisAlignedHypercube.h"

/*! \class BoundingBox
 *  \brief A BoundingBox bounds the region of space occupied by some object.  It is axis-aligned.
 */
class BoundingBox
  : public AxisAlignedHypercube<Point, Point::Scalar, 3>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand
     */
    typedef AxisAlignedHypercube<Point, Point::Scalar, 3> Parent;

    /*! Null constructor puts the min corner at +infinity and the max corner at -infinity.
     */
    inline BoundingBox(void);

    /*! Constructor accepts a min corner and a max corner.
     *  \param min Sets mMinCorner.
     *  \param max Sets mMaxCorner.
     */
    inline BoundingBox(const Point &min, const Point &max);

    /*! This method intersects a Ray against this BoundingBox.
     *  \param r The Ray to intersect.
     *  \param t0 If an intersection exists, the parameter value at the first intersection with r is returned here.
     *  \param t1 If an intersections exists, the parameter value at the second intersection with r is returned here.
     *  \return true if an intersection exists; false, otherwise.
     *  \note t0's and t1's values are undefined after this method exits if the corresponding intersection did not exist.
     */
    bool intersect(const Ray &r, float &t0, float &t1) const;

    /*! This method intersects a Ray against this BoundingBox.
     *  \param r The Ray to intersect.
     *  \param t If an intersection exists, the parameter value at the first intersection with r is returned here.
     *  \param n If an intersection exists, the BoundingBox's Normal at the intersection with r is returned here.
     *  \return true if an intersection exists; false, otherwise.
     *  \note t's and n's values are undefined after this method exits if an intersection did not exist.
     */
    bool intersect(const Ray &r, float &t, Normal &n) const;
}; // end class BoundingBox

#include "BoundingBox.inl"

#endif // BOUNDING_BOX_H

