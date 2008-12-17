/*! \file Vector2.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a N=2 specialization
 *         of Vector.
 */

#pragma once

#include "Vector.h"

namespace gpcpu
{

template<typename S>
  class Vector<S,2>
{
  public:
    /*! \typedef This
     *  \brief Shorthand.
     */
    typedef Vector<S,2> This;

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
        :x(v[0]),y(v[1]){}

    /*! \fn Vector
     *  \brief Constructor takes a const pointer to
     *         an 2-length array of Scalars.
     *  \param v A pointer to an N-length array of
     *           Scalars to copy from.
     */
    inline Vector(const Scalar *v)
      :x(v[0]),y(v[1]){}

    /*! \fn Vector
     *  \brief This method sets every element of this Vector
     *         to the given value.
     *  \param v The fill value.
     */
    inline Vector(const Scalar v)
      :x(v),y(v){}

    /*! \fn Vector
     *  \brief Special constructor for 3-vectors.
     *  \param s0 The first element.
     *  \param s1 The second element.
     *  \param s2 The third element.
     */
    inline Vector(const Scalar &v0,
                  const Scalar &v1)
      :x(v0),y(v1){}

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
      {return i == 0 ? x : y;}

    /*! \fn operator[]
     *  \brief This method provides const access to the ith element.
     *  \return A reference to the ith element.
     */
    template<typename IndexType>
      inline const Scalar &operator[](const IndexType &i) const
      {return i == 0 ? x : y;}

    template<size_t i>
      inline Scalar &get(void)
      {return i == 0 ? x : y;}

    template<size_t i>
      inline const Scalar &get(void) const
      {return i == 0 ? x : y;}

    /*! \fn operator+
     *  \brief Addition operator.
     *  \return Returns (*this) + rhs
     */
    inline This operator+(const This &rhs) const
      {return This(x + rhs.x, y + rhs.y);}

    /*! \fn operator+=
     *  \brief Plus equal operator.
     *  \param rhs The right hand side of the relation.
     *  \return *this
     */
    inline This &operator+=(const This &rhs)
      {x += rhs.x; y += rhs.y; return *this;}

    /*! \fn operator*=
     *  \brief Scalar times equal.
     *  \param s The Scalar to multiply by.
     *  \return *this.
     */
    inline This &operator*=(const Scalar &s)
      {x *= s; y *= s; return *this;}

    /*! \fn operator/
     *  \brief Scalar divide.
     *  \param rhs The Scalar to divide by.
     *  \return (*this) / s
     */
    inline This operator/(const Scalar &rhs) const
      {return This(x / rhs, y / rhs);}

    /*! \fn operator/=
     *  \brief Scalar divide equal.
     *  \param rhs The right hand side of the operation.
     *  \return (*this)
     */
    inline This &operator/=(const Scalar &rhs)
      {x /= rhs; y /= rhs; return *this;}

    /*! \fn operator/=
     *  \brief Vector component-wise divide equal.
     *  \param rhs The vector to divide by.
     *  \return (*this) / rhs
     */
    inline This &operator/=(const Vector &rhs)
      {x /= rhs.x; y /= rhs.y; return *this;}

    /*! \fn operator/
     *  \brief Vector component-wise divide.
     *  \param rhs The vector to divide by.
     *  \return (*this) / rhs
     */
    inline This operator/(const Vector &rhs) const
      {return This(x / rhs.x, y / rhs.y);}

    /*! \fn operator*
     *  \brief Vector component-wise mutliply
     *  \param rhs The vector to multiply by.
     *  \return (*this) * rhs
     */
    inline This operator*(const This &rhs) const
      {return This(x * rhs.x, y * rhs.y);}

    /*! \fn operator*
     *  \brief Scalar multiply.
     *  \param rhs The Scalar to multiply by.
     *  \return (*this) * rhs
     */
    inline This operator*(const Scalar &rhs) const
      {return This(x * rhs, y * rhs);}

    /*! \fn operator*
     *  \brief Vector component-wise mutliply equal.
     *  \param rhs The vector to multiply by.
     *  \return *this. 
     */
    inline This &operator*=(const This &rhs)
      {x *= rhs.x; y *= rhs.y; return *this;}

    /*! \fn dot
     *  \brief Dot product
     *  \param rhs The vector to dot by.
     *  \return (*this) dot rhs
     */
    inline Scalar dot(const This &rhs) const
      {return x*rhs.x + y*rhs.y;}

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
       y = __gpcpu_max<Scalar>(0, __gpcpu_min<Scalar>(1, y));}

    /*! \fn posDot
     *  \brief Dot product where negative values
     *         are clamped to 0.
     *  \param rhs The vector to dot by.
     *  \return max(0, (*this) dot rhs)
     */
    inline Scalar posDot(const This &rhs) const
      {return __gpcpu_max<Scalar>(0, dot(rhs));}

    /*! \fn operator-
     *  \brief Unary negation.
     *  \return -(*this)
     */
    inline This operator-(void) const
      {return This(-x,-y);}

    /*! \fn operator-
     *  \brief Binary minus.
     *  \param rhs The right hand side of the operation.
     *  \return (*this) - rhs
     */
    inline This operator-(const This &rhs) const
      {return This(x - rhs.x, y - rhs.y);}

    /*! \fn operator-=
     *  \brief Decrement equal.
     *  \param rhs The right hand side of the operation.
     *  \return *this
     */
    inline This &operator-=(const This &rhs)
      {x -= rhs.x; y -= rhs.y; return *this;}

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
      {return x + y;}

    /*! \fn product
     *  \brief This method returns the product of elements of
     *         this Vector.
     *  \return As above.
     */
    inline Scalar product(void) const
      {return x * y;}

    /*! \fn max
     *  \brief This method returns the maximum element of this
     *         Vector.
     *  \return The maximum element of this Vector.
     */
    inline Scalar maxElement(void) const
      {return __gpcpu_max<Scalar>(x, y);}

    /*! \fn maxElementIndex
     *  \brief This method returns the index of the maximum element
     *         of this Vector.
     *  \return The index of the maximum element of this Vector.
     */
    inline size_t maxElementIndex(void) const
    {
      if(x > y)
      {
        return 0;
      } // end if

      return 1;
    } // end maxElementIndex()

    /*! \fn mean
     *  \brief This method returns the mean of the elements of
     *         this Vector.
     *  \return The mean of this Vector's elements.
     */
    inline Scalar mean(void) const
      {return (x + y) * mOneOverN;}

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
      {return 2;}

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

    const static Scalar mOneOverN;
}; // end Vector

// initialize the const static member outside the class
template<typename S>
  const S Vector<S,2>::mOneOverN = S(1) / 2;

} // end gpcpu

