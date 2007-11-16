/*! \file RandomAccessFilm.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RandomAccessFilm class.
 */

#include "RandomAccessFilm.h"

#ifdef WIN32
#define OPENEXR_DLL
#endif // WIN32

#include <halfLimits.h>
#include <ImfRgbaFile.h>
#include <ImfRgba.h>
#include <ImfStringAttribute.h>
#include <ImfHeader.h>
#include <ImfArray.h>

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
  ::raster(const size_t px,
           const size_t py)
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
  // map (u,v) to an integer grid location and
  // a fractional delta
  size_t i, j;
  float uDel, vDel;
  column(u, i, uDel);
  row(v, j, vDel);

  // i+1 & j+1 will be clamped
  // XXX TODO implement this correctly with boundary conditions
  const Pixel &ul = raster(i, std::min(mDimensions[1]-1, j+1));
  const Pixel &ur = raster(std::min(mDimensions[0]-1, i+1), std::min(mDimensions[1]-1, j+1));
  const Pixel &ll = raster(i, j);
  const Pixel &lr = raster(std::min(mDimensions[0]-1, i+1), j);

  float oneMinusUDel = 1.0f - uDel;
  float oneMinusVDel = 1.0f - vDel;

  return oneMinusUDel*oneMinusVDel*ll
       +         uDel*oneMinusVDel*lr
       + oneMinusUDel*        vDel*ul
       +         uDel*        vDel*ur;
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

void RandomAccessFilm
  ::tonemap(void)
{
  // compute compute world adaptation luminance Ywa
  // as the log average over all pixels' luminance
  float Ywa = 0;
  for(size_t i = 0; i < size(); ++i)
  {
    float l = (*this)[i].luminance();
    if(l > 0) Ywa += logf(l);
  } // end for i

  Ywa = expf(Ywa / size());
  float invYwa2 = 1.0f / (Ywa * Ywa);

  for(size_t i = 0; i < size(); ++i)
  {
    // shorthand
    Pixel &p = (*this)[i];

    // compute luminance
    float lum = p.luminance();

    p *= (1.0f + lum * invYwa2);
    p /= (1.0f + lum);
  } // end for i
} // end RandomAccessFilm::tonemap()

RandomAccessFilm &RandomAccessFilm
  ::operator-=(const RandomAccessFilm &rhs)
{
  if(rhs.getWidth() == getWidth() && rhs.getHeight() == getHeight())
  {
    for(size_t i = 0; i < size(); ++i)
    {
      (*this)[i] -= rhs[i];
    } // end for i
  } // end if

  return *this;
} // end RandomAccessFilm::operator-=()

void RandomAccessFilm
  ::getRasterPosition(const float u, const float v,
                      size_t &i, size_t &j) const
{
  i = column(u);
  j = row(v);
} // end RandomAccessFilm::getRasterPosition()

void RandomAccessFilm
  ::readEXR(const char *filename)
{
  Imf::RgbaInputFile file(filename);

  Imath::Box2i dw = file.dataWindow();

  unsigned int w = dw.max.x - dw.min.x + 1;
  unsigned int h = dw.max.y - dw.min.y + 1;

  Imf::Array2D<Imf::Rgba> pixels;
  pixels.resizeErase(h, w);

  file.setFrameBuffer(&pixels[0][0] - dw.min.x - dw.min.y * w, 1, w);
  file.readPixels(dw.min.y, dw.max.y);

  // now resize ourself
  resize(w,h);

  for(size_t y = 0; y < getHeight(); ++y)
  {
    for(size_t x = 0; x < getWidth(); ++x)
    {
      // flip the image because EXR is stored ass-backwards
      raster(x,y)[0] = pixels[h-1-y][x].r;
      raster(x,y)[1] = pixels[h-1-y][x].g;
      raster(x,y)[2] = pixels[h-1-y][x].b;
    } // end for
  } // end for
} // end RandomAccessFilm::readEXR()

void RandomAccessFilm
  ::writeEXR(const char *filename) const
{
  size_t w = getWidth();
  size_t h = getHeight();

  // convert image to rgba
  std::vector<Imf::Rgba> pixels(getWidth() * getHeight());
  for(unsigned int y = 0; y < getHeight(); ++y)
  {
    for(unsigned int x = 0; x < getWidth(); ++x)
    {
      // flip the image because EXR is stored ass-backwards
      pixels[(h-1-y)*w + x].r = raster(x,y)[0];
      pixels[(h-1-y)*w + x].g = raster(x,y)[1];
      pixels[(h-1-y)*w + x].b = raster(x,y)[2];
      pixels[(h-1-y)*w + x].a = 1.0f;
    } // end for
  } // end for

  Imf::Header header(getWidth(), getHeight());
  header.insert("renderer", Imf::StringAttribute("Gotham"));

  Imf::RgbaOutputFile file(filename, header, Imf::WRITE_RGBA);
  file.setFrameBuffer(&pixels[0], 1, getWidth());

  file.writePixels(getHeight());
} // end RandomAccessFilm::writeEXR()

