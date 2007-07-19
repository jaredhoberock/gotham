/*! \file Vector.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a 3-vector class.
 */

#ifndef VECTOR
#define VECTOR

#include <gpcpu/Vector.h>

class Vector
  : public gpcpu::float3
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef gpcpu::float3 Parent;

    /*! Null constructor calls the Parent.
     */
    inline Vector(void);

    /*! Copy constructor accepts a Parent.
     *  \param v The Parent to copy from.
     */
    inline Vector(const Parent &v);

    /*! Constructor accepts x-, y-, and z- coordinates.
     *  \param x The x-coordinate.
     *  \param y The y-coordinate.
     *  \param z The z-coordinate.
     */
    inline Vector(const float x, const float y, const float z);
}; // end Vector

// XXX remove this
typedef Vector Vector3;

#include "Vector.inl"

#endif // VECTOR

