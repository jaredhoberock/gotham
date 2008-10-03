/*! \file RandomAccessFilm.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RandomAccessFilm class.
 */

#ifdef WIN32
#define OPENEXR_DLL
#define NOMINMAX
#endif // WIN32

#include "RandomAccessFilm.h"
#include <thresholdversusintensity/thresholdVersusIntensity.h>
#include <algorithm>

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

float RandomAccessFilm
  ::computeSumLuminance(void) const
{
  float result = 0;
  for(size_t i = 0; i < size(); ++i)
  {
    result += (*this)[i].luminance();
  } // end for i

  return result;
} // end RandomAccessFilm::computeSumLuminance()

float RandomAccessFilm
  ::computeSumLogLuminance(void) const
{
  float result = 0;
  for(size_t i = 0; i < size(); ++i)
  {
    if((*this)[i].luminance() > 0)
      result += logf((*this)[i].luminance());
  } // end for i

  return result;
} // end RandomAccessFilm::computeSumLuminance()

RandomAccessFilm::Pixel RandomAccessFilm
  ::computeMean(void) const
{
  return computeSum() / (getWidth() * getHeight());
} // end RandomAccessFilm::computeMean()

float RandomAccessFilm
  ::computeMeanLuminance(void) const
{
  return computeSumLuminance() / (getWidth() * getHeight());
} // end RandomAccessFilm::computeMeanLuminance()

float RandomAccessFilm
  ::computeMinLuminance(void) const
{
  float result = std::numeric_limits<float>::infinity();

  for(size_t y = 0; y != getHeight(); ++y)
  {
    for(size_t x = 0; x != getWidth(); ++x)
    {
      result = std::min(result, raster(x,y).luminance());
    } // end for x
  } // end for y

  return result;
} // end RandomAccessFilm::computeMinLuminance()

float RandomAccessFilm
  ::computeMaxLuminance(void) const
{
  float result = -std::numeric_limits<float>::infinity();

  for(size_t y = 0; y != getHeight(); ++y)
  {
    for(size_t x = 0; x != getWidth(); ++x)
    {
      result = std::max(result, raster(x,y).luminance());
    } // end for x
  } // end for y

  return result;
} // end RandomAccessFilm::computeMaxLuminance()

float RandomAccessFilm
  ::computeMedianLuminance(void) const
{
  std::vector<float> l;
  for(size_t i = 0; i != size(); ++i)
    l.push_back((*this)[i].luminance());

  std::sort(l.begin(), l.end());

  return l[l.size()/2];
} // end RandomAccessFilm::computeMedianLuminance()

float RandomAccessFilm
  ::computeMeanLuminanceIgnoreZero(void) const
{
  float result = 0;
  float l;
  size_t n = 0;
  for(size_t i = 0; i != size(); ++i)
  {
    l = (*this)[i].luminance();
    if(l > 0)
    {
      result += l;
      ++n;
    } // end if
  } // end for i

  if(n > 0)
  {
    result /= n;
  } // end if

  return result;
} // end RandomAccessFilm::computeMeanLuminanceIgnoreZero()

float RandomAccessFilm
  ::computeMeanLogLuminance(void) const
{
  return computeSumLogLuminance() / (getWidth() * getHeight());
} // end RandomAccessFilm::computeMeanLogLuminance()

float RandomAccessFilm
  ::computeVarianceLuminance(void) const
{
  float mean = computeMeanLuminance();

  float result = 0;
  for(size_t i = 0; i != size(); ++i)
  {
    float diff = mean - (*this)[i].luminance();
    result += diff * diff;
  } // end for i

  result /= (getWidth() * getHeight());

  return result;
} // end RandomAccessFilm::computeVarianceLuminance()

float RandomAccessFilm
  ::computeVarianceLuminanceIgnoreZero(void) const
{
  float mean = computeMeanLuminanceIgnoreZero();

  float result = 0;
  float l = 0;
  size_t n = 0;
  for(size_t i = 0; i != size(); ++i)
  {
    l = (*this)[i].luminance();
    if(l > 0)
    {
      float diff = mean - l;
      result += diff * diff;
      ++n;
    } // end if
  } // end for i

  if(n > 0)
  {
    result /= n;
  } // end if

  return result;
} // end RandomAccessFilm::computeVarianceLuminance()

float RandomAccessFilm
  ::computeVarianceLogLuminance(void) const
{
  float mean = computeMeanLogLuminance();

  float result = 0;
  for(size_t i = 0; i != size(); ++i)
  {
    if((*this)[i].luminance() > 0)
    {
      float diff = mean - logf((*this)[i].luminance());
      result += diff * diff;
    } // end if
  } // end for i

  result /= (getWidth() * getHeight());

  return result;
} // end RandomAccessFilm::computeVarianceLogLuminance()

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
  ::erode(const float h)
{
  size_t holesLeft = 0;
  size_t holesFixed = 0;

  RandomAccessFilm copy = *this;

  for(size_t y = 0; y != getHeight(); ++y)
  {
    for(size_t x = 0; x != getWidth(); ++x)
    {
      for(size_t c = 0; c != Spectrum::numElements(); ++c)
      {
        if(raster(x,y)[c] == h)
        {
          // visit each non-hole neighbor
          float newVal = 0;

          float n = 0;
          float neighbors = 0;

          // top left
          if(x > 0 && y + 1 < getHeight())
          {
            n = raster(x-1,y+1)[c];
            if(n != h)
            {
              newVal += n;
              ++neighbors;
            } // end if
          } // end if

          // top
          if(y + 1 < getHeight())
          {
            n = raster(x,y+1)[c];
            if(n != h)
            {
              newVal += n;
              ++neighbors;
            } // end if
          } // end if

          // top right
          if(x + 1 < getWidth() && y + 1 < getHeight())
          {
            n = raster(x+1,y+1)[c];
            if(n != h)
            {
              newVal += n;
              ++neighbors;
            } // end if
          } // end if

          // left
          if(x > 0)
          {
            n = raster(x-1,y)[c];
            if(n != h)
            {
              newVal += n;
              ++neighbors;
            } // end if
          } // end if

          // right
          if(x + 1 < getWidth())
          {
            n = raster(x+1,y)[c];
            if(n != h)
            {
              newVal += n;
              ++neighbors;
            } // end if
          } // end if

          // bottom left
          if(x > 0 && y > 0)
          {
            n = raster(x-1,y-1)[c];
            if(n != h)
            {
              newVal += n;
              ++neighbors;
            } // end if
          } // end if

          // bottom
          if(y > 0)
          {
            n = raster(x,y-1)[c];
            if(n != h)
            {
              newVal += n;
              ++neighbors;
            } // end if
          } // end if

          // bottom right
          if(x + 1 < getWidth() && y > 0)
          {
            n = raster(x+1,y-1)[c];
            if(n != h)
            {
              newVal += n;
              ++neighbors;
            } // end if
          } // end if

          if(neighbors > 0)
          {
            newVal /= neighbors;
            copy.raster(x,y)[c] = newVal;
            ++holesFixed;
          } // end if
          else
          {
            ++holesLeft;
          } // end else
        } // end if
      } // end for c
    } // end for x

    *this = copy;
  } // end for y

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
        float sDiff = (gpcpu::float2(x,y) - gpcpu::float2(x,wy)).norm();

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
        float sDiff = (gpcpu::float2(x,y) - gpcpu::float2(wx,y)).norm();

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

void RandomAccessFilm
  ::applyClamp(const float m, const float M)
{
  for(size_t y = 0; y != getHeight(); ++y)
  {
    for(size_t x = 0; x != getWidth(); ++x)
    {
      Pixel &p = raster(x,y);
      p[0] = std::max(m, std::min(p[0], M));
      p[1] = std::max(m, std::min(p[1], M));
      p[2] = std::max(m, std::min(p[2], M));
    } // end for x
  } // end for y
} // end RandomAccessFilm::applyClamp()

void RandomAccessFilm
  ::applyGammaAndExposure(const float gamma, const float exposure)
{
  float invGamma = 1.0f / gamma;
  float scale = powf(2.0f, exposure);

  for(size_t y = 0; y != getHeight(); ++y)
  {
    for(size_t x = 0; x != getWidth(); ++x)
    {
      Pixel &p = raster(x,y);
      p[0] = powf(scale * p[0], invGamma);
      p[1] = powf(scale * p[1], invGamma);
      p[2] = powf(scale * p[2], invGamma);
    } // end for x
  } // end for y
} // end RandomAccessFilm::applyGammaAndExposure()

void RandomAccessFilm
  ::applySqrt(void)
{
  for(size_t y = 0; y != getHeight(); ++y)
  {
    for(size_t x = 0; x != getWidth(); ++x)
    {
      Pixel &p = raster(x,y);
      p[0] = sqrtf(std::max(0.0f,p[0]));
      p[1] = sqrtf(std::max(0.0f,p[1]));
      p[2] = sqrtf(std::max(0.0f,p[2]));
    } // end for x
  } // end for y
} // end RandomAccessFilm::applySqrt()

void RandomAccessFilm
  ::applyPow(const float e)
{
  for(size_t y = 0; y != getHeight(); ++y)
  {
    for(size_t x = 0; x != getWidth(); ++x)
    {
      Pixel &p = raster(x,y);
      p[0] = powf(std::max(0.0f,p[0]), e);
      p[1] = powf(std::max(0.0f,p[1]), e);
      p[2] = powf(std::max(0.0f,p[2]), e);
    } // end for x
  } // end for y
} // end RandomAccessFilm::applySqrt()

void RandomAccessFilm
  ::divideLuminance(const RandomAccessFilm &rhs, const float epsilon)
{
  for(size_t y = 0; y != getHeight(); ++y)
  {
    for(size_t x = 0; x != getWidth(); ++x)
    {
      Pixel &p = raster(x,y);

      float r = 1.0f / std::max(rhs.raster(x,y).luminance(), epsilon);
      p[0] *= r;
      p[1] *= r;
      p[2] *= r;
    } // end for x
  } // end for y
} // end RandomAccessFilm::divideLuminance()

float RandomAccessFilm
  ::computeLuminancePercentile(const float p) const
{
  std::vector<float> luminance;

  // make a list of luminance
  for(size_t i = 0; i != size(); ++i)
  {
    luminance.push_back((*this)[i].luminance());
  } // end for i

  // sort
  std::sort(luminance.begin(), luminance.end());

  return luminance[std::min<size_t>(size(), static_cast<size_t>(p * size()))];
} // end RandomAccessFilm::computeLuminancePercentile()

float RandomAccessFilm
  ::computeLuminancePercentileIgnoreZero(const float p) const
{
  std::vector<float> luminance;

  // make a list of luminance
  for(size_t i = 0; i != size(); ++i)
  {
    float l = (*this)[i].luminance();

    if(l > 0
       && l == l
       && l != std::numeric_limits<float>::infinity())
    {
      luminance.push_back((*this)[i].luminance());
    } // end if
  } // end for i

  // sort
  std::sort(luminance.begin(), luminance.end());

  float result = 0;
  if(luminance.size() > 0)
  {
    size_t index = std::min<size_t>(luminance.size()-1,
                                    static_cast<size_t>(p * luminance.size()));
    result = luminance[index];
  } // end if

  if(result == 0)
  {
    std::cerr << "RandomAccessFilm::computeLuminancePercentileIgnoreZero(): Failed." << std::endl;
  } // end if

  return result;
} // end RandomAccessFilm::computeLuminancePercentile()

void RandomAccessFilm
  ::clampLuminance(const float m, const float M)
{
  for(size_t i = 0; i != size(); ++i)
  {
    Pixel &p = (*this)[i];

    float l = p.luminance();

    if(l < m)
    {
      p.setLuminance(m);
    } // end if
    else if(l > M)
    {
      p.setLuminance(M);
    } // end else if
  } // end for i
} // end RandomAccessFilm::clampLuminance()

void RandomAccessFilm
  ::applyThresholdVersusIntensityFunction(void)
{
  float maxDisplay = 100.0f;

  for(size_t y = 0; y != getHeight(); ++y)
  {
    for(size_t x = 0; x != getWidth(); ++x)
    {
      Spectrum &p = raster(x,y);

      // compute the luminance of p
      float L = p.luminance();

      // map to display luminance
      L *= maxDisplay;

      float t = tvi(L);

      raster(x,y) = Spectrum::white();
      raster(x,y).setLuminance(t);
    } // end for x
  } // end for y
} // end RandomAccessFilm::applyThresholdVersusIntensityFunction()

