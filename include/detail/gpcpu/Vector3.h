/*! \file Vector3.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a N=3 specialization
 *         of Vector.
 */

#pragma once

#include "Vector.h"

namespace gpcpu
{

template<typename S>
  class Vector<S,3>
{
  public:
    /*! \typedef This
     *  \brief Shorthand.
     */
    typedef Vector<S,3> This;

    /*! \typedef Scalar
     *  \brief Shorthand.
     */
    typedef S Scalar;

    /*! \fn Vector
     *  \brief Null constructor does nothing.
     */
    inline Vector(void){};

    /*! \fn Vector
     *  \brief Templated universal copy constructor
     *         accepts anything that can be indexed.
     *  \param v Something to copy from.
     */
    template<typename CopyFromType>
      inline Vector(const CopyFromType &v)
        :x(v[0]),y(v[1]),z(v[2]){}

    /*! \fn Vector
     *  \brief Constructor takes a const pointer to
     *         an 3-length array of Scalars.
     *  \param v A pointer to an N-length array of
     *           Scalars to copy from.
     */
    inline Vector(const Scalar *v)
      :x(v[0]),y(v[1]),z(v[2]){}

    /*! \fn Vector
     *  \brief This method sets every element of this Vector
     *         to the given value.
     *  \param v The fill value.
     */
    inline Vector(const Scalar v)
      :x(v),y(v),z(v){}

    /*! \fn Vector
     *  \brief Special constructor for 3-vectors.
     *  \param s0 The first element.
     *  \param s1 The second element.
     *  \param s2 The third element.
     */
    inline Vector(const Scalar &v0,
                  const Scalar &v1,
                  const Scalar &v2)
      :x(v0),y(v1),z(v2){}

    /*! \fn Vector
     *  \brief Constructor accepts a smaller Vector
     *         and a final Scalar.
     *  \param v The first N-1 Scalars.
     *  \param s The Nth Scalar.
     */
    inline Vector(const Vector<Scalar,3-1> &v,
                  const Scalar &s)
      :x(v[0]),y(v[1]),z(s){}

    /*! \fn operator Scalar *
     *  \brief Cast to Scalar * operator.
     *  \return Returns &x.
     */
    inline operator Scalar *(void)
      {return &x;}

    /*! \fn operator const Scalar * ()
     *  \brief Cast to const Scalar * operator.
     *  \return Returns &x.
     */
    inline operator const Scalar *(void) const
      {return &x;}

    /*! \fn operator[]
     *  \brief This method provides access to the ith element.
     *  \return A reference to the ith element.
     */
    template<typename IndexType>
      inline Scalar &operator[](const IndexType &i)
      {return i == 0 ? x : i == 1 ? y : z;}

    /*! \fn operator[]
     *  \brief This method provides const access to the ith element.
     *  \return A reference to the ith element.
     */
    template<typename IndexType>
      inline const Scalar &operator[](const IndexType &i) const
      {return i == 0 ? x : i == 1 ? y : z;}

    template<size_t i>
      inline Scalar &get(void)
      {return i == 0 ? x : i == 1 ? y : z;}

    template<size_t i>
      inline const Scalar &get(void) const
      {return i == 0 ? x : i == 1 ? y : z;}

    /*! \fn operator+
     *  \brief Addition operator.
     *  \return Returns (*this) + rhs
     */
    inline This operator+(const This &rhs) const
      {return This(x + rhs.x, y + rhs.y, z + rhs.z);}

    /*! \fn operator+=
     *  \brief Plus equal operator.
     *  \param rhs The right hand side of the relation.
     *  \return *this
     */
    inline This &operator+=(const This &rhs)
      {x += rhs.x; y += rhs.y; z += rhs.z; return *this;}

    /*! \fn operator*=
     *  \brief Scalar times equal.
     *  \param s The Scalar to multiply by.
     *  \return *this.
     */
    inline This &operator*=(const Scalar &s)
      {x *= s; y *= s; z *= s; return *this;}

    /*! \fn operator/
     *  \brief Scalar divide.
     *  \param rhs The Scalar to divide by.
     *  \return (*this) / s
     */
    inline This operator/(const Scalar &rhs) const
      {return This(x / rhs, y / rhs, z / rhs);}

    /*! \fn operator/=
     *  \brief Scalar divide equal.
     *  \param rhs The right hand side of the operation.
     *  \return (*this)
     */
    inline This &operator/=(const Scalar &rhs)
      {x /= rhs; y /= rhs; z /= rhs; return *this;}

    /*! \fn operator/=
     *  \brief Vector component-wise divide equal.
     *  \param rhs The vector to divide by.
     *  \return (*this) / rhs
     */
    inline This &operator/=(const Vector &rhs)
      {x /= rhs.x; y /= rhs.y; z /= rhs.z; return *this;}

    /*! \fn operator/
     *  \brief Vector component-wise divide.
     *  \param rhs The vector to divide by.
     *  \return (*this) / rhs
     */
    inline This operator/(const Vector &rhs) const
      {return This(x / rhs.x, y / rhs.y, z / rhs.z);}

    /*! \fn operator*
     *  \brief Vector component-wise mutliply
     *  \param rhs The vector to multiply by.
     *  \return (*this) * rhs
     */
    inline This operator*(const This &rhs) const
      {return This(x * rhs.x, y * rhs.y, z * rhs.z);}

    /*! \fn operator*
     *  \brief Scalar multiply.
     *  \param rhs The Scalar to multiply by.
     *  \return (*this) * rhs
     */
    inline This operator*(const Scalar &rhs) const
      {return This(x * rhs, y * rhs, z * rhs);}

    /*! \fn operator*
     *  \brief Vector component-wise mutliply equal.
     *  \param rhs The vector to multiply by.
     *  \return *this. 
     */
    inline This &operator*=(const This &rhs)
      {x *= rhs.x; y *= rhs.y; z *= rhs.z; return *this;}

    /*! \fn dot
     *  \brief Dot product
     *  \param rhs The vector to dot by.
     *  \return (*this) dot rhs
     */
    inline Scalar dot(const This &rhs) const
      {return x*rhs.x + y*rhs.y + z*rhs.z;}

    /*! \fn absDot
     *  \brief Absolute value of dot product.
     *  \param rhs The vector to dot by.
     *  \return |(*this) dot rhs)|
     */
    inline Scalar absDot(const This &rhs) const
      {return fabs(dot(rhs));}

    /*! \fn saturate
     *  \brief This method clamps this Vector's elements
     *         to [0,1].
     */
    inline void saturate(void)
      {x = __gpcpu_max<Scalar>(0, __gpcpu_min<Scalar>(1, x));
       y = __gpcpu_max<Scalar>(0, __gpcpu_min<Scalar>(1, y));
       z = __gpcpu_max<Scalar>(0, __gpcpu_min<Scalar>(1, z));}

    /*! \fn posDot
     *  \brief Dot product where negative values
     *         are clamped to 0.
     *  \param rhs The vector to dot by.
     *  \return max(0, (*this) dot rhs)
     */
    inline Scalar posDot(const This &rhs) const
      {return __gpcpu_max<Scalar>(0, dot(rhs));}

    /*! \fn cross
     *  \brief Cross product
     *  \param rhs The rhs of the operation.
     *  \return (*this) cross rhs
     *  \note This in only implemented for 3-vectors.
     */
    inline This cross(const This &rhs) const
    {
      This result;
      const This &lhs = *this;

      This subtractMe(lhs.z*rhs.y, lhs.x*rhs.z, lhs.y*rhs.x);

      result.x = (lhs.y * rhs.z);
      result.x -= subtractMe.x;
      result.y = (lhs.z * rhs.x);
      result.y -= subtractMe.y;
      result.z = (lhs.x * rhs.y);
      result.z -= subtractMe.z;

      return result;
    } // end cross()

    /*! \fn operator-
     *  \brief Unary negation.
     *  \return -(*this)
     */
    inline This operator-(void) const
      {return This(-x,-y,-z);}

    /*! \fn operator-
     *  \brief Binary minus.
     *  \param rhs The right hand side of the operation.
     *  \return (*this) - rhs
     */
    inline This operator-(const This &rhs) const
      {return This(x - rhs.x, y - rhs.y, z - rhs.z);}

    /*! \fn operator-=
     *  \brief Decrement equal.
     *  \param rhs The right hand side of the operation.
     *  \return *this
     */
    inline This &operator-=(const This &rhs)
      {x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this;}

    /*! \fn norm
     *  \brief This method returns the 2-norm of this
     *         Vector.
     *  \return As above.
     */
    inline Scalar norm(void) const
      {return Sqrt<Scalar>()(norm2());}

    /*! \fn length
     *  \brief Alias for norm().
     *  \return norm().
     */
    inline Scalar length(void) const
      {return norm();}

    /*! \fn norm2
     *  \brief This method returns the squared 2-norm of
     *         this Vector.
     *  \return As above.
     */
    inline Scalar norm2(void) const
      {return dot(*this);}

    /*! \fn sum
     *  \brief This method returns the sum of elements of
     *         this Vector.
     *  \return As above.
     */
    inline Scalar sum(void) const
      {return x + y + z;}

    /*! \fn product
     *  \brief This method returns the product of elements of
     *         this Vector.
     *  \return As above.
     */
    inline Scalar product(void) const
      {return x * y * z;}

    /*! \fn max
     *  \brief This method returns the maximum element of this
     *         Vector.
     *  \return The maximum element of this Vector.
     */
    inline Scalar maxElement(void) const
      {return __gpcpu_max<Scalar>(x, __gpcpu_max<Scalar>(y, z));}

    /*! \fn maxElementIndex
     *  \brief This method returns the index of the maximum element
     *         of this Vector.
     *  \return The index of the maximum element of this Vector.
     */
    inline size_t maxElementIndex(void) const
    {
      if(x > y)
      {
        if(x > z) return 0;
        else return 2;
      } // end if
      else if(y > z) return 1;
      return 2;
    } // end maxElementIndex()

    /*! \fn mean
     *  \brief This method returns the mean of the elements of
     *         this Vector.
     *  \return The mean of this Vector's elements.
     */
    inline Scalar mean(void) const
      {return (x + y + z) * mOneOverN;}

    /*! \fn normalize
     *  \brief This method returns a normalized version
     *         of this Vector.
     *  \return (*this) / this->norm()
     */
    inline This normalize(void) const
      { return *this / norm();}

    /*! This method returns the dimension of this Vector.
     *  \return N
     */
    inline static size_t numElements(void)
      {return 3;}

    /*! This method returns a Vector orthogonal to this
     *  Vector
     *  \return A Vector v such that this->dot(v) is small.
     */
    This orthogonalVector(void) const
    {
      int i = -1, j = -1, k = -1;

      // choose the minimal element
      if(fabs(x) <= fabs(y))
      {
        if(fabs(x) <= fabs(y))
        {
          // x is min
          k = 0;
          i = 1;
          j = 2;
        } // end if
        else
        {
          // z is min
          k = 2;
          i = 0;
          j = 1;
        } // end else
      } // end if
      else if(fabs(y) <= fabs(z))
      {
        // y is min
        k = 1;
        i = 0;
        j = 2;
      } // end else if
      else
      {
        // z is min
        k = 2;
        i = 0;
        j = 1;
      } // end else

      // supposing that y was the min, result would look like:
      // result = (z / sqrt(1.0 - y*y), 0, -x / sqrt(1.0 - y*y))
      Scalar denom = Sqrt<Scalar>()(Scalar(1.0)- (*this)[k]*(*this)[k]);

      This result;
      result[i] =  (*this)[j] / denom;
      result[j] = -(*this)[i] / denom;
      result[k] = 0;

      return result;
    } // end orthogonalVector()

    /*! This method reflects this Vector about another.
     *  \param v The Vector about which to reflect.
     *  \return The reflection of this Vector about v.
     */
    This reflect(const Vector &v) const
    {
      This result = Scalar(2) * this->dot(v)*(*this) - v;
      return result;
    } // end reflect()

    Scalar x;
    Scalar y;
    Scalar z;

    const static Scalar mOneOverN;
}; // end Vector

// initialize the const static member outside the class
template<typename S>
  const S Vector<S,3>::mOneOverN = S(1) / 3;

/*! \fn cross
 *  \brief cross product for Vector3s.
 *  \param lhs The left hand side of the cross product.
 *  \param rhs The right hand side of the cross product.
 *  \return lhs x rhs
 */
template<typename Scalar>
  inline Vector<Scalar,3> cross(const Vector<Scalar,3> &lhs, const Vector<Scalar,3> &rhs)
{
  return lhs.cross(rhs);
} // end cross()

} // end gpcpu

