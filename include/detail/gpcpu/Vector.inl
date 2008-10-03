/*! \file Vector.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Vector.h.
 */

#include "Vector.h"
#include <math.h>
#include <limits>

namespace gpcpu
{

template<typename T>
  inline T __gpcpu_max(const T &lhs, const T &rhs)
{
  return lhs > rhs ? lhs : rhs;
} // end __gpcpu_max()

template<typename T>
  inline T __gpcpu_min(const T &lhs, const T &rhs)
{
  return lhs < rhs ? lhs : rhs;
} // end __gpcpu_min()

template<typename Scalar, size_t N>
  Vector<Scalar, N>
    ::Vector(void)
{
  ;
} // end Vector::Vector()

template<typename Scalar, size_t N>
  template<typename CopyFromType>
    Vector<Scalar, N>
      ::Vector(const CopyFromType &v)
{
  for(size_t i = 0; i < N; ++i)
    mElements[i] = v[i];
} // end Vector::Vector()

template<typename Scalar, size_t N>
  Vector<Scalar, N>
    ::Vector(const Scalar v)
{
  for(size_t i = 0; i < N; ++i)
    mElements[i] = v;
} // end Vector::Vector()

template<typename Scalar, size_t N>
  Vector<Scalar, N>
    ::Vector(const Scalar *v)
{
  memcpy(mElements, v, N*sizeof(Scalar));
} // end Vector::Vector()

template<typename Scalar, size_t N>
  Vector<Scalar, N>
    ::Vector(const Scalar &v0,
             const Scalar &v1)
{
  mElements[0] = v0;
  mElements[1] = v1;
} // end Vector::Vector()

template<typename Scalar, size_t N>
  Vector<Scalar, N>
    ::Vector(const Scalar &v0,
             const Scalar &v1,
             const Scalar &v2,
             const Scalar &v3)
{
  mElements[0] = v0;
  mElements[1] = v1;
  mElements[2] = v2;
  mElements[3] = v3;
} // end Vector::Vector()

template<typename Scalar, size_t N>
  Vector<Scalar, N>
    ::Vector(const Scalar &v0,
             const Scalar &v1,
             const Scalar &v2,
             const Scalar &v3,
             const Scalar &v4)
{
  mElements[0] = v0;
  mElements[1] = v1;
  mElements[2] = v2;
  mElements[3] = v3;
  mElements[4] = v4;
} // end Vector::Vector()

template<typename Scalar, size_t N>
  Vector<Scalar, N>
    ::Vector(const Vector<Scalar,N-1> &v,
             const Scalar &s)
{
  size_t i = 0;
  for(; i < N-1; ++i)
    mElements[i] = v[i];

  mElements[i] = s;
} // end Vector::Vector()

template<typename Scalar, size_t N>
  template<typename IndexType>
    Scalar &Vector<Scalar,N>
      ::operator[](const IndexType &i)
{
  return mElements[i];
} // end Vector::operator[]()

template<typename Scalar, size_t N>
  template<typename IndexType>
    const Scalar &Vector<Scalar,N>
     ::operator[](const IndexType &i) const
{
  return mElements[i];
} // end Vector::operator[]()

template<typename Scalar, size_t N>
  Vector<Scalar,N>
    ::operator Scalar * ()
{
  return mElements;
} // end Vector::operator Scalar * ()

template<typename Scalar, size_t N>
  Vector<Scalar, N>
    ::operator const Scalar * () const
{
  return mElements;
} // end Vector::operator Scalar * ()

template<typename Scalar, size_t N>
  Vector<Scalar,N> &Vector<Scalar, N>
    ::operator*=(const Scalar &s)
{
  for(size_t i = 0; i < N; ++i)
  {
    mElements[i] *= s;
  } // end for i

  return *this;
} // end Vector::operator*=()

template<typename Scalar, size_t N>
  Vector<Scalar,N> Vector<Scalar, N>
    ::operator*(const Scalar &rhs) const
{
  This result;
  for(size_t i = 0; i < N; ++i)
  {
    result[i] = mElements[i] * rhs;
  } // end for i

  return result;
} // end Vector::operator*()

template<typename Scalar, size_t N>
  Vector<Scalar,N> Vector<Scalar, N>
    ::operator*(const This &rhs) const
{
  This result;
  for(size_t i = 0; i < N; ++i)
  {
    result[i] = mElements[i] * rhs[i];
  } // end for i

  return result;
} // end Vector::operator*()

template<typename Scalar, size_t N>
  Vector<Scalar,N> &Vector<Scalar, N>
    ::operator*=(const This &rhs)
{
  for(size_t i = 0; i < N; ++i)
  {
    mElements[i] *= rhs[i];
  } // end for i

  return *this;
} // end Vector::operator*=()

template<typename Scalar, size_t N>
  Scalar Vector<Scalar, N>
    ::dot(const This &rhs) const
{
  Scalar result = 0;
  Scalar temp;
  for(size_t i = 0; i < N; ++i)
  {
    temp = (*this)[i] * rhs[i];
    result += temp;
  } // end for i

  return result;
} // end Vector::dot()

template<typename Scalar, size_t N>
  Scalar Vector<Scalar, N>
    ::absDot(const This &rhs) const
{
  return fabs(dot(rhs));
} // end Vector::absDot()

template<typename Scalar, size_t N>
  Scalar Vector<Scalar, N>
    ::posDot(const This &rhs) const
{
  return __gpcpu_max<Scalar>(0, dot(rhs));
} // end Vector::posDot()

template<typename Scalar, size_t N>
  void Vector<Scalar, N>
    ::saturate(void)
{
  for(size_t i = 0; i != N; ++i)
  {
    (*this)[i] = __gpcpu_max<Scalar>(0, __gpcpu_min<Scalar>(1, (*this)[i]));
  } // end for i
} // end Vector::saturate()

template<typename Scalar, size_t N>
  Vector<Scalar, N> Vector<Scalar, N>
    ::operator+(const This &rhs) const
{
  This result;
  for(size_t i = 0; i < N; ++i)
  {
    result[i] = this->operator[](i) + rhs.operator[](i);
  } // end for i

  return result;
} // end Vector::operator+()

template<typename Scalar, size_t N>
  Vector<Scalar, N> &Vector<Scalar, N>
    ::operator+=(const This &rhs)
{
  for(size_t i = 0; i < N; ++i)
  {
    this->operator[](i) += rhs.operator[](i);
  } // end for i

  return *this;
} // end Vector::operator+()

template<typename Scalar, size_t N>
  Vector<Scalar, N> Vector<Scalar, N>
   ::operator/(const Scalar &rhs) const
{
  This result;
  for(size_t i = 0; i < N; ++i)
  {
    result[i] = (*this)[i] / rhs;
  } // end for i

  return result;
} // end Vector::operator/()

template<typename Scalar, size_t N>
  Vector<Scalar, N> &Vector<Scalar, N>
    ::operator/=(const This &rhs)
{
  This &result = *this;
  for(size_t i = 0; i < N; ++i)
  {
    result[i] = (*this)[i] / rhs[i];
  } // end for i

  return result;
} // end Vector::operator/=()

template<typename Scalar, size_t N>
  Vector<Scalar, N> Vector<Scalar, N>
    ::operator/(const This &rhs) const
{
  This result = *this;
  return result /= rhs;
} // end Vector::operator/=()

template<typename Scalar, size_t N>
  Vector<Scalar, N> Vector<Scalar, N>
    ::operator-(void) const
{
  This result;
  for(size_t i = 0; i < N; ++i)
  {
    result[i] = -(*this)[i];
  } // for i

  return result;
} // end Vector::operator-()

template<typename Scalar, size_t N>
  Vector<Scalar, N> Vector<Scalar, N>
    ::operator-(const This &rhs) const
{
  This result;
  for(size_t i = 0; i < N; ++i)
  {
    result[i] = (*this)[i] - rhs[i];
  } // end for i

  return result;
} // end Vector::operator-()

template<typename Scalar, size_t N>
  Scalar Vector<Scalar, N>
    ::norm(void) const
{
  return Sqrt<Scalar>()(norm2());
} // end Vector::norm()

template<typename Scalar, size_t N>
  Scalar Vector<Scalar, N>
    ::length(void) const
{
  return norm();
} // end Vector::length()

template<typename Scalar, size_t N>
  Scalar Vector<Scalar, N>
    ::sum(void) const
{
  Scalar result = Scalar(0);
  for(size_t i = 0; i < N; ++i)
  {
    result += (*this)[i];
  } // end for i

  return result;
} // end Vector::sum()

template<typename Scalar, size_t N>
  Scalar Vector<Scalar, N>
    ::product(void) const
{
  Scalar result = Scalar(1);
  for(size_t i = 0; i < N; ++i)
  {
    result *= (*this)[i];
  } // end for i

  return result;
} // end Vector::product()

template<typename Scalar, size_t N>
  Scalar Vector<Scalar, N>
    ::norm2(void) const
{
  Scalar result = 0.0;
  for(size_t i = 0; i < N; ++i)
  {
    result += (*this)[i] * (*this)[i];
  } // end for i

  return result;
} // end Vector::norm2()

template<typename Scalar, size_t N>
  Vector<Scalar, N> Vector<Scalar, N>
    ::normalize(void) const
{
  return (*this) / norm();
} // end Vector::normalize()

template<typename Scalar, size_t N>
  Vector<Scalar, N> &Vector<Scalar, N>
    ::operator/=(const Scalar &rhs)
{
  for(size_t i = 0; i < N; ++i)
  {
    (*this)[i] /= rhs;
  } // end for i

  return *this;
} // end for Vector::operator/=()

template<typename Scalar, size_t N>
  Vector<Scalar, N> &Vector<Scalar, N>
    ::operator-=(const This &rhs)
{
  for(size_t i = 0; i < N; ++i)
  {
    (*this)[i] -= rhs[i];
  } // end for i

  return *this;
} // end Vector::operator-=()

template<typename Scalar, size_t N>
  Vector<Scalar, N>
    operator*(const Scalar &lhs, const Vector<Scalar,N> &rhs)
{
  Vector<Scalar, N> result;
  for(size_t i = 0; i < N; ++i)
  {
    result[i] = lhs * rhs[i];
  } // end for i

  return result;
} // end operator*()

template<typename Scalar, size_t N>
  Scalar dot(const Vector<Scalar,N> &lhs, const Vector<Scalar,N> &rhs)
{
  return lhs.dot(rhs);
} // end dot()

template<typename Scalar, size_t N>
  Vector<Scalar,N> saturate(const Vector<Scalar,N> &v)
{
  Vector<Scalar,N> result = v;
  result.saturate();
  return result;
} // end dot()

template<typename Scalar, size_t N>
  Vector<Scalar,N> normalize(const Vector<Scalar,N> &v)
{
  return v.normalize();
} // end normalize()

template<typename Scalar, size_t N>
  Vector<Scalar,N> reflect(const Vector<Scalar,N> &ref,
                           const Vector<Scalar,N> &v)
{
  return ref.reflect(v);
} // end reflect()

#ifndef __CUDACC__
template<typename Scalar, size_t N>
  std::ostream &operator<<(std::ostream &os,
                           const Vector<Scalar,N> &v)
{
  size_t i = 0;
  for(; i < N - 1; ++i)
  {
    os << v[i] << " ";
  } // end for i

  os << v[i];

  return os;
} // end operator<<()

template<typename Scalar, size_t N>
  std::istream &operator>>(std::istream &is,
                           Vector<Scalar,N> &v)
{
  for(size_t i = 0; i < N; ++i)
  {
    is >> v[i];
  } // end for i

  return is;
} // end operator>>()
#endif // __CUDACC__

template<typename Scalar, size_t N>
  size_t Vector<Scalar,N>
    ::numElements(void)
{
  return N;
} // end Vector::numElements()

template<typename Scalar, size_t N>
  Scalar Vector<Scalar, N>
    ::maxElement(void) const
{
  Scalar result = -std::numeric_limits<Scalar>::infinity();
  for(size_t i = 0; i < N; ++i)
  {
    result = mElements[i] > result ? mElements[i] : result;
  } // end for

  return result;
} // end Vector::maxElement()

template<typename Scalar, size_t N>
  size_t Vector<Scalar, N>
    ::maxElementIndex(void) const
{
  size_t index = N;
  Scalar result = -std::numeric_limits<Scalar>::infinity();
  for(size_t i = 0; i < N; ++i)
  {
    if(mElements[i] > result)
    {
      result = mElements[i];
      index = i;
    } // end if
  } // end for

  return index;
} // end Vector::maxElementIndex()

template<typename Scalar, size_t N>
  const Scalar Vector<Scalar,N>::mOneOverN = static_cast<Scalar>(1.0) / N;

template<typename Scalar, size_t N>
  Scalar Vector<Scalar, N>
    ::mean(void) const
{
  Scalar result = 0;
  for(size_t i = 0; i < N; ++i)
  {
    result += mElements[i];
  } // end for

  return result * mOneOverN;
} // end Vector::mean()

template<typename Scalar, size_t N>
  typename Vector<Scalar, N>::This Vector<Scalar, N>
    ::reflect(const This &v) const
{
  This result = Scalar(2) * this->dot(v)*(*this) - v; 
  return result;
} // end Vector::reflect()

} // end namespace gpcpu

