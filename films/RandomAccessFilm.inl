/*! \file RandomAccessFilm.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RandomAccessFilm class.
 */

#include "RandomAccessFilm.h"

RandomAccessFilm
  ::RandomAccessFilm(void)
    :Parent0(),Parent1()
{
  ;
} // end RandomAccessFilm::RandomAccessFilm()

RandomAccessFilm
  ::RandomAccessFilm(const unsigned int width,
                     const unsigned int height)
    :Parent0(width,height),Parent1(width,height)
{
  ;
} // end RandomAccessFilm::RandomAccessFilm()

RandomAccessFilm::Pixel &RandomAccessFilm
  ::pixel(const float u,
          const float v)
{
  return Parent1::element(u,v);
} // end RandomAccessFilm::pixel()

const RandomAccessFilm::Pixel &RandomAccessFilm
  ::pixel(const float u,
          const float v) const
{
  return Parent1::element(u,v);
} // end RandomAccessFilm::pixel()

RandomAccessFilm::Pixel &RandomAccessFilm
  ::raster(const unsigned int px,
           const unsigned int py)
{
  return Parent1::raster(px,py);
} // end RandomAccessFilm::raster()

const RandomAccessFilm::Pixel &RandomAccessFilm
  ::raster(const unsigned int px,
           const unsigned int py) const
{
  return Parent1::raster(px,py);
} // end RandomAccessFilm::raster()

RandomAccessFilm::Pixel RandomAccessFilm
  ::bilerp(const float u,
           const float v) const
{
  float uFloor = floorf(u);
  float uCeil = ceilf(u);
  float vFloor = floorf(v);
  float vCeil = ceilf(v);

  const Pixel &ul = pixel(uFloor, vCeil);
  const Pixel &ur = pixel(uCeil,  vCeil);
  const Pixel &ll = pixel(uFloor, vFloor);
  const Pixel &lr = pixel(uCeil,  vFloor);

  float uDel = u - uFloor;
  float oneMinusUDel = 1.0f - uDel;
  Pixel top    = oneMinusUDel * ul + uDel * ur;
  Pixel bottom = oneMinusUDel * ll + uDel * lr;

  float vDel = v - vFloor;
  float oneMinusVDel = 1.0f - vDel;
  return oneMinusVDel * bottom + vDel * top;
} // end RandomAccessFilm::bilerp()

void RandomAccessFilm
  ::fill(const Pixel &v)
{
  std::fill(begin(), end(), v);
} // end RandomAccessFilm::fill()

void RandomAccessFilm
  ::scale(const Pixel &s)
{
  for(size_t i = 0; i < size(); ++i)
  {
    (*this)[i] *= s;
  } // end i
} // end RandomAccessFilm::scale()

void RandomAccessFilm
  ::resize(const unsigned int width,
           const unsigned int height)
{
  Parent0::resize(width,height);
  Parent1::resize(width,height);
} // end RandomAccessFilm::resize()

RandomAccessFilm::Pixel RandomAccessFilm
  ::computeSum(void) const
{
  Pixel result(0,0,0);
  for(size_t i = 0; i < size(); ++i)
  {
    result += (*this)[i];
  } // end for i

  return result;
} // end RandomAccessFilm::computeSum()

RandomAccessFilm::Pixel RandomAccessFilm
  ::computeMean(void) const
{
  return computeSum() / (getWidth() * getHeight());
} // end RandomAccessFilm::computeMean()

