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
  size_t xClamp = std::min<size_t>(px, getWidth() - 1);
  size_t yClamp = std::min<size_t>(py, getHeight() - 1);

  return Parent1::raster(xClamp,yClamp);
} // end RandomAccessFilm::raster()

const RandomAccessFilm::Pixel &RandomAccessFilm
  ::raster(const size_t px,
           const size_t py) const
{
  size_t xClamp = std::min<size_t>(px, getWidth() - 1);
  size_t yClamp = std::min<size_t>(py, getHeight() - 1);

  return Parent1::raster(xClamp,yClamp);
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

void RandomAccessFilm
  ::resample(RandomAccessFilm &target) const
{
  gpcpu::float2 targetStep(1.0f / target.getWidth(),
                           1.0f / target.getHeight());

  // for every pixel of the target
  Pixel integral;
  for(size_t ty = 0; ty < target.getHeight(); ++ty)
  {
    for(size_t tx = 0; tx < target.getWidth(); ++tx)
    {
      // find the rectangle in the [0,1)^2 to integrate 
      gpcpu::float2 rectStart(tx,ty);
      rectStart *= targetStep;

      gpcpu::float2 rectEnd = rectStart;
      rectEnd += targetStep;

      // we need to integrate this rectangle in the source
      integrateRectangle(rectStart[0], rectStart[1],
                         rectEnd[0], rectEnd[1],
                         integral);

      target.raster(tx,ty) = integral;
    } // end for tx
  } // end for ty
} // end RandomAccessFilm::resample()

void RandomAccessFilm
  ::integrateRectangle(const float xStart, const float yStart,
                       const float xEnd, const float yEnd,
                       Pixel &integral) const
{
  // this implementation is nasty, but I think it is correct

  integral = Spectrum::black();

  // find the step size of this image
  gpcpu::float2 step(1.0f / getWidth(), 1.0f / getHeight());

  // get the raster indices of the lower left pixel
  gpcpu::size2 ll;
  getRasterPosition(xStart, yStart, ll[0], ll[1]);

  // get the raster indices of the upper right pixel
  gpcpu::size2 ur;
  getRasterPosition(xEnd, yEnd, ur[0], ur[1]);

  size_t x = ll[0];
  size_t y = ll[1];
  
  // integrate the lower left pixel
  // the width of the lower left pixel in the unit square
  // is min((x+1)*step[0], xEnd) - xStart
  // the height of the lower left pixel in the unit square
  // is min((y+1)*step[1], yEnd) - yStart
  // the mins account for the case where either dimension of the rectangle is
  // is smaller than a pixel
  float leftColumnWidth = std::min(static_cast<float>(x+1)*step[0], xEnd) - xStart;
  float bottomRowHeight = std::min(static_cast<float>(y+1)*step[1], yEnd) - yStart;
  integral += leftColumnWidth*bottomRowHeight * raster(x, y);

  // now integrate the bottom partial row
  for(x = x + 1; x < ur[0]; ++x)
  {
    // the width of these pixels simply the step size
    integral += step[0] * bottomRowHeight * raster(x, y);
  } // end for x

  // calculate the width of the right column
  // this width is negative if there is no right column
  float rightColumnWidth = xEnd - step[0] * static_cast<float>(ur[0]);

  // integrate the lower right pixel (if there is one)
  if(ll[0] != ur[0])
  {
    integral += rightColumnWidth*bottomRowHeight * raster(ur[0], y);
  } // end if




  // now integrate full rows
  // pixels within a full row have area equal to step.product()
  float innerPixelsArea = step.product();
  for(y = ll[1] + 1; y < ur[1]; ++y)
  {
    x = ll[0];

    // integrate the left pixel
    integral += leftColumnWidth * step[1] * raster(x, y);

    // now integrate the inner row
    for(x = x + 1; x < ur[0]; ++x)
    {
      integral += innerPixelsArea * raster(x,y);
    } // end for x

    // now integrate the right pixel (if there is one)
    if(ll[0] != ur[0])
    {
      integral += rightColumnWidth * step[1] * raster(ur[0], y);
    } // end if
  } // end for

  

  // now integrate the top partial row (if these exist)
  if(ll[1] != ur[1])
  {
    y = ur[1];
    x = ll[0];

    // calculate the height of the top row
    // this height is negative if there is no top row
    float topRowHeight = yEnd - step[1] * static_cast<float>(ur[1]);

    // integrate the upper left pixel
    integral += leftColumnWidth * topRowHeight * raster(x,y);

    // now integrate the top partial row
    for(x = x + 1; x < ur[0]; ++x)
    {
      // the width of these pixels simply the step size
      integral += step[0] * topRowHeight * raster(x, y);
    } // end for x

    // now integrate the right pixel (if there is one)
    if(ll[0] != ur[0])
    {
      integral += rightColumnWidth*topRowHeight * raster(ur[0], y);
    } // end if
  } // end if

  // now divide by the area of integration
  integral /= ((xEnd-xStart) * (yEnd - yStart));
} // end RandomAccessFilm::integrateRectangle()

size_t RandomAccessFilm
  ::erode(const Pixel &h)
{
  size_t holesLeft = 0;
  size_t holesFixed = 0;

  for(size_t y = 0; y != getHeight(); ++y)
  {
    for(size_t x = 0; x != getWidth(); ++x)
    {
      if(raster(x,y) == h)
      {
        // visit each non-hole neighbor
        Pixel newVal(Pixel(0,0,0));

        float neighbors = 0;

        // top left
        if(x > 0 && y + 1 < getHeight())
        {
          newVal += raster(x-1,y+1);
          ++neighbors;
        } // end if

        // top
        if(y + 1 < getHeight())
        {
          newVal += raster(x,y+1);
          ++neighbors;
        } // end if

        // top right
        if(x + 1 < getWidth() && y + 1 < getHeight())
        {
          newVal += raster(x+1,y+1);
          ++neighbors;
        } // end if

        // left
        if(x > 0)
        {
          newVal += raster(x-1,y);
          ++neighbors;
        } // end if

        // right
        if(x + 1 < getWidth())
        {
          newVal += raster(x+1,y);
          ++neighbors;
        } // end if

        // bottom left
        if(x > 0 && y > 0)
        {
          newVal += raster(x-1,y-1);
          ++neighbors;
        } // end if

        // bottom
        if(y > 0)
        {
          newVal += raster(x,y-1);
          ++neighbors;
        } // end if

        // bottom right
        if(x + 1 < getWidth() && y > 0)
        {
          newVal += raster(x+1,y-1);
          ++neighbors;
        } // end if

        if(neighbors > 0)
        {
          newVal /= neighbors;
          raster(x,y) = newVal;
          ++holesFixed;
        } // end if
        else
        {
          ++holesLeft;
        } // end else
      } // end if
    } // end for x
  } // end for y

  std::cerr << "RandomAccessFilm::erode(): Fixed " << holesFixed << " holes." << std::endl;

  return holesLeft;
} // end RandomAccessFilm::erode()

inline static float square(const float v)
{
  return v*v;
} // end square()

void RandomAccessFilm
  ::bilateralFilter(const float sigmad,
                    const float sigmar,
                    const RandomAccessFilm &intensity)
{
  if(intensity.getWidth() != getWidth() ||
     intensity.getHeight() != getHeight())
  {
    std::cerr << "RandomAccessFilm::bilateralFilter(): intensity must be same resolution as this!" << std::endl;
    return;
  } // end if

  // clamp the search radius to 3*sigmad
  size_t radius = static_cast<size_t>(ceilf(3.0f * sigmad));

  // precompute these divisions
  float invSigmad = 1.0f / sigmad;
  float invSigmar = 1.0f / sigmar;

  using namespace gpcpu;

  // separable convolution
  RandomAccessFilm firstHalf = *this;
  for(size_t y = 0; y != intensity.getHeight(); ++y)
  {
    for(size_t x = 0; x != intensity.getWidth(); ++x)
    {
      float sumOfWeights = 0;
      Spectrum blurred(0,0,0);

      for(size_t wy = std::max<int>(0, y - radius);
          wy != std::min<size_t>(intensity.getHeight(), y + radius + 1);
          ++wy)
      {
        // get a diff in intensity domain
        float iDiff = (intensity.raster(x,y) - intensity.raster(x,wy)).norm();

        // get a diff in spacial domain
        float sDiff = (float2(x,y) - float2(x,wy)).norm();

        // compute weight
        float w = expf(-0.5f*square(iDiff*invSigmar))
                * expf(-0.5f*square(sDiff*invSigmad));

        // sum weight
        sumOfWeights += w;

        // sum into result
        blurred += w * raster(x,wy);
      } // end for wy

      // now normalize
      firstHalf.raster(x,y) = blurred / sumOfWeights;
    } // end for x
  } // end for y

  RandomAccessFilm result = *this;
  for(size_t y = 0; y != intensity.getHeight(); ++y)
  {
    for(size_t x = 0; x != intensity.getWidth(); ++x)
    {
      float sumOfWeights = 0;
      Spectrum blurred(0,0,0);

      for(size_t wx = std::max<int>(0, x - radius);
          wx != std::min<size_t>(intensity.getWidth(), x + radius + 1);
          ++wx)
      {
        // get a diff in intensity domain
        float iDiff = (intensity.raster(x,y) - intensity.raster(wx,y)).norm();

        // get a diff in spacial domain
        float sDiff = (float2(x,y) - float2(wx,y)).norm();

        // compute weight
        float w = expf(-0.5f*square(iDiff*invSigmar))
                * expf(-0.5f*square(sDiff*invSigmad));

        // sum weight
        sumOfWeights += w;

        // sum into result
        blurred += w * firstHalf.raster(wx,y);
      } // end for wy

      // now normalize
      result.raster(x,y) = blurred / sumOfWeights;
    } // end for x
  } // end for y

  *this = result;
} // end RandomAccessFilm::bilateralFilter()

