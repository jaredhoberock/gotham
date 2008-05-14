/*! \file Array2.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Array2.h.
 */

#include "Array2.h"
#include <assert.h>
#include <bittricks/bittricks.h>

template<typename Type>
  Array2<Type>
    ::Array2(void)
      :Parent(),mDimensions(0,0)
{
  ;
} // end Array2::Array2()

template<typename Type>
  Array2<Type>
    ::Array2(const size_t width,
             const size_t height)
      :Parent(width*height)
{
  mDimensions[0] = static_cast<unsigned int>(width);
  mDimensions[1] = static_cast<unsigned int>(height);
} // end Array2::Array2()

template<typename Type>
  size_t Array2<Type>
    ::column(const float t) const
{
  return ifloor(t * mDimensions[0]);
} // end Array2::column()

template<typename Type>
  void Array2<Type>
    ::column(float t, size_t &i, float &frac) const
{
  t *= mDimensions[0];
  i = ifloor(t);
  frac = t - i;
} // end Array2::column()

template<typename Type>
  size_t Array2<Type>
    ::row(const float t) const
{
  return ifloor(t * mDimensions[1]);
} // end Array::row()

template<typename Type>
  void Array2<Type>
    ::row(float t, size_t &i, float &frac) const
{
  t *= mDimensions[1];
  i = ifloor(t);
  frac = t - i;
} // end Array2::row()

template<typename Type>
  typename Array2<Type>::Element &Array2<Type>
    ::element(const float u,
              const float v)
{
  size_t ei = column(u);
  size_t ej = row(v);

  ei = std::min<size_t>(ei, mDimensions[0] - 1);
  ej = std::min<size_t>(ej, mDimensions[1] - 1);

  return raster(ei,ej);
} // end Array2::element()

template<typename Type>
  const typename Array2<Type>::Element &Array2<Type>
    ::element(const float u,
              const float v) const
{
  size_t ei = column(u);
  size_t ej = row(v);

  ei = std::min<size_t>(ei, mDimensions[0] - 1);
  ej = std::min<size_t>(ej, mDimensions[1] - 1);

  return raster(ei,ej);
} // end Array2::element()

template<typename Type>
  typename Array2<Type>::Element &Array2<Type>
    ::raster(const size_t i,
             const size_t j)
{
  assert(j*mDimensions[0] + i < Parent::size());
  return (*this)[j*mDimensions[0] + i];
} // end Array2::raster()

template<typename Type>
  typename Array2<Type>::size2 Array2<Type>
    ::rasterCoordinates(const float u,
                        const float v) const
{
  return size2(static_cast<size_t>(u * mDimensions[0]),
               static_cast<size_t>(v * mDimensions[1]));
} // end Array2::rasterCoordinates()

template<typename Type>
  const typename Array2<Type>::Element &Array2<Type>
    ::raster(const size_t i,
             const size_t j) const
{
  assert(j*mDimensions[0] + i < Parent::size());
  return (*this)[j*mDimensions[0] + i];
} // end Array2::raster()

template<typename Type>
  void Array2<Type>
    ::fill(const Element &v)
{
  std::fill(Parent::begin(), Parent::end(), v);
} // end Array2::fill()

template<typename Type>
  void Array2<Type>
    ::resize(const size_t width,
             const size_t height)
{
  mDimensions[0] = static_cast<unsigned int>(width);
  mDimensions[1] = static_cast<unsigned int>(height);
  Parent::resize(width*height);
} // end Array2::resize()

template<typename Type>
  const typename Array2<Type>::size2 &Array2<Type>
    ::getDimensions(void) const
{
  return mDimensions;
} // end Array2::getDimensions()

