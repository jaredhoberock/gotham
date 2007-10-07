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

void RandomAccessFilm
  ::fill(const Pixel &v)
{
  std::fill(begin(), end(), v);
} // end RandomAccessFilm::fill()

void RandomAccessFilm
  ::resize(const unsigned int width,
           const unsigned int height)
{
  Parent0::resize(width,height);
  Parent1::resize(width,height);
} // end RandomAccessFilm::resize()

