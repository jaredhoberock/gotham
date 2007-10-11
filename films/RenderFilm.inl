/*! \file RenderFilm.inl
 *  \author Jared Hoberock
 *  \brief Inline file for RenderFilm.h.
 */

#include "RenderFilm.h"

#ifdef WIN32
#define OPENEXR_DLL
#endif // WIN32

#include <halfLimits.h>
#include <ImfRgbaFile.h>
#include <ImfRgba.h>
#include <ImfStringAttribute.h>
#include <ImfHeader.h>

RenderFilm
  ::RenderFilm(void)
    :Parent(),mFilename(""),mNormalizeOnPostprocess(false)
{
  ;
} // end RenderFilm::RenderFilm()

RenderFilm
  ::RenderFilm(const unsigned int width,
               const unsigned int height,
               const std::string &filename)
    :Parent(width,height),mFilename(filename),mNormalizeOnPostprocess(false)
{
  reset();
} // end RenderFilm::RenderFilm()

void RenderFilm
  ::resize(const unsigned int width,
           const unsigned int height)
{
  Parent::resize(width,height);
  reset();
} // end RenderFilm::resize()

void RenderFilm
  ::reset(void)
{
  mNumDeposits = 0;
  mSum = Spectrum::black();
  mMaximumLuminance = 0;
  mMinimumLuminance = 0;
} // end RenderFilm::reset()

void RenderFilm
  ::deposit(const float px, const float py,
            const Spectrum &s)
{
  Spectrum &p = Parent::pixel(px,py);
  p += s;
  mSum += s;
  ++mNumDeposits;

  if(p.luminance() > mMaximumLuminance)
  {
    mMaximumLuminance = p.luminance();
  } // end if
  if(p.luminance() < mMinimumLuminance)
  {
    mMinimumLuminance = p.luminance();
  } // end if
} // end RenderFilm::deposit()

void RenderFilm
  ::deposit(const size_t rx, const size_t ry,
            const Spectrum &s)
{
  Spectrum &p = Parent::raster(rx,ry);
  p += s;
  mSum += s;
  ++mNumDeposits;

  if(p.luminance() > mMaximumLuminance)
  {
    mMaximumLuminance = p.luminance();
  } // end if
} // end RenderFilm::deposit()

const Spectrum &RenderFilm
  ::getSum(void) const
{
  return mSum;
} // end RenderFilm::getSum()

size_t RenderFilm
  ::getNumDeposits(void) const
{
  return mNumDeposits;
} // end RenderFilm::getNumDeposits()

void RenderFilm
  ::fill(const Pixel &v)
{
  Parent::fill(v);
  reset();
  mMaximumLuminance = mMinimumLuminance = v.luminance();
  mSum = static_cast<float>(getWidth() * getHeight()) * v;
} // end RenderFilm::fill()

void RenderFilm
  ::scale(const Pixel &s)
{
  Parent::scale(s);
  mMaximumLuminance *= s.luminance();
  mMinimumLuminance *= s.luminance();
  mSum *= s;
} // end RenderFilm::scale()

float RenderFilm
  ::getMaximumLuminance(void) const
{
  return mMaximumLuminance;
} // end RenderFilm::getMaximumLuminance()

float RenderFilm
  ::getMinimumLuminance(void) const
{
  return mMinimumLuminance;
} // end RenderFilm::getMinimumLuminance()

const RandomAccessFilm::Pixel &RenderFilm
  ::pixel(const float u, const float v) const
{
  return Parent::pixel(u,v);
} // end RenderFilm::pixel()

void RenderFilm
  ::postprocess(void)
{
  if(mNormalizeOnPostprocess)
  {
    // scale by 1/mMaximumLuminance
    float s = 1.0f / getMaximumLuminance();
    scale(Spectrum(s,s,s));
  } // end if

  if(mFilename != "")
  {
    writeEXR(mFilename.c_str());
  } // end if
} // end RenderFilm::postprocess()

const std::string &RenderFilm
  ::getFilename(void) const
{
  return mFilename;
} // end RenderFilm::getFilename()

void RenderFilm
  ::setFilename(const std::string &filename)
{
  mFilename = filename;
} // end RenderFilm::setFilename()

void RenderFilm
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
} // end RenderFilm::writeEXR()

void RenderFilm
  ::init(void)
{
  ;
} // end RenderFilm::init()

void RenderFilm
  ::setNormalizeOnPostprocess(const bool n)
{
  mNormalizeOnPostprocess = n;
} // end RenderFilm::setNormalizeOnPostprocess()

