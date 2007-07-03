/*! \file Point.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class abstracting a point in 3D.
 */

#ifndef POINT_H
#define POINT_H

#include <gpcpu/Vector.h>

/*! \class Point
 *  \brief A Point is a 3-vector.
 */
class Point : public float3
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef float3 Parent;

    /*! Null constructor calls the parent.
     */
    inline Point(void);

    /*! Copy constructor accepts a Parent.
     *  \param p Sets this Point.
     */
    inline Point(const Parent &p);

    /*! Constructor accepts 3 coordinates.
     *  \param x Sets the x coordinate.
     *  \param y Sets the y coordinate.
     *  \param z Sets the z coordinate.
     */
    inline Point(const float x, const float y, const float z);

    /*! This static method returns a Point at infinity.
     *  \return (infinity, infinity, infinity)
     */
    inline static Point infinity(void);
}; // end class Point

#include "Point.inl"

#endif // POINT_H

