/*! \file RenderFilm.inl
 *  \author Jared Hoberock
 *  \brief Inline file for RenderFilm.h.
 */

#include "RenderFilm.h"
#include "../path/PathToImage.h"

RenderFilm
  ::RenderFilm(void)
    :Parent0(),Parent1(),mFilename(""),mApplyTonemap(false)
{
  ;
} // end RenderFilm::RenderFilm()

RenderFilm
  ::RenderFilm(const unsigned int width,
               const unsigned int height,
               const std::string &filename)
    :Parent0(),Parent1(width,height),mFilename(filename),mApplyTonemap(false)
{
  reset();
} // end RenderFilm::RenderFilm()

void RenderFilm
  ::resize(const unsigned int width,
           const unsigned int height)
{
  Parent1::resize(width,height);
  reset();
} // end RenderFilm::resize()

void RenderFilm
  ::reset(void)
{
  mNumDeposits = 0;
  mSum = Spectrum::black();
  mSumLogLuminance = 0;
  mMaximumLuminance = 0;
  mMinimumLuminance = 0;
} // end RenderFilm::reset()

void RenderFilm
  ::deposit(const float px, const float py,
            const Spectrum &s)
{
  unsigned int rx, ry;
  getRasterPosition(px, py, rx, ry);
  deposit(rx, ry, s);
} // end RenderFilm::deposit()

void RenderFilm
  ::deposit(const size_t rx, const size_t ry,
            const Spectrum &s)
{
  Spectrum &p = Parent1::raster(rx,ry);

  float oldLum = p.luminance();
  if(oldLum > 0)
  {
    mSumLogLuminance -= logf(oldLum);
  } // end if

  p += s;
  mSum += s;
  ++mNumDeposits;

  float lum = p.luminance();

  if(lum > mMaximumLuminance)
  {
    mMaximumLuminance = p.luminance();
  } // end if

  if(lum > 0)
  {
    mSumLogLuminance += logf(lum);
  } // end if
} // end RenderFilm::deposit()

const Spectrum &RenderFilm
  ::getSum(void) const
{
  return mSum;
} // end RenderFilm::getSum()

float RenderFilm
  ::getSumLogLuminance(void) const
{
  return mSumLogLuminance;
} // end RenderFilm::getSumLogY()

size_t RenderFilm
  ::getNumDeposits(void) const
{
  return mNumDeposits;
} // end RenderFilm::getNumDeposits()

void RenderFilm
  ::fill(const Pixel &v)
{
  Parent1::fill(v);
  reset();
  float lum = v.luminance();
  mMaximumLuminance = mMinimumLuminance = lum;
  mSum = static_cast<float>(getWidth() * getHeight()) * v;

  if(lum > 0)
  {
    mSumLogLuminance += static_cast<float>(getWidth() * getHeight()) * logf(lum);
  } // end if
  else
  {
    mSumLogLuminance = 0;
  } // end else
} // end RenderFilm::fill()

void RenderFilm
  ::scale(const Pixel &s)
{
  Parent1::scale(s);
  float lum = s.luminance();
  mMaximumLuminance *= lum;
  mMinimumLuminance *= lum;
  mSum *= s;

  // we have to recompute mSumLogLuminance
  mSumLogLuminance = 0;
  for(size_t j = 0; j < getHeight(); ++j)
  {
    for(size_t i = 0; i < getWidth(); ++i)
    {
      float lum = raster(i,j).luminance();
      if(lum > 0)
      {
        mSumLogLuminance += logf(lum);
      } // end if
    } // end for i
  } // end for j
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
  return Parent1::pixel(u,v);
} // end RenderFilm::pixel()

const RandomAccessFilm::Pixel &RenderFilm
  ::raster(const unsigned int i, const unsigned int j) const
{
  return Parent1::raster(i,j);
} // end RenderFilm::raster()

void RenderFilm
  ::preprocess(void)
{
  // preprocess the parent first
  Parent0::preprocess();

  // init to black
  fill(Spectrum::black());
} // end RenderFilm::preprocess()

void RenderFilm
  ::postprocess(void)
{
  // postprocess the parent first
  Parent0::postprocess();

  // tonemap?
  if(mApplyTonemap) tonemap();

  if(mFilename != "")
  {
    writeEXR(mFilename.c_str());
    std::cout << "Wrote result to: " << getFilename() << std::endl;
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
  ::init(void)
{
  ;
} // end RenderFilm::init()

void RenderFilm
  ::setApplyTonemap(const bool a)
{
  mApplyTonemap = a;
} // end RenderFilm::setApplyTonemap()

void RenderFilm
  ::record(const float w,
           const PathSampler::HyperPoint &x,
           const Path &xPath,
           const std::vector<PathSampler::Result> &results)
{
  // XXX TODO make this a member of this class
  // RATIONALE: could want to generalize the class to something that
  //            has a state -- we could implement different kinds of
  //            cameras this way
  PathToImage mapToImage;

  Spectrum d;
  for(size_t i = 0;
      i != results.size();
      ++i)
  {
    // shorthand
    const PathSampler::Result &r = results[i];

    // map the result to a location in the image
    gpcpu::float2 pixel;
    mapToImage(r, x, xPath, pixel[0], pixel[1]);

    // each deposit takes the following form:
    d = (w * r.mWeight / r.mPdf) * r.mThroughput;

    // handoff to deposit() 
    deposit(pixel[0], pixel[1], d);
  } // end for r
} // end RenderFilm::record()

void RenderFilm
  ::recordSquare(const float w,
                 const PathSampler::HyperPoint &x,
                 const Path &xPath,
                 const std::vector<PathSampler::Result> &results)
{
  // XXX TODO implement this correctly for Paths which may generate
  //          several results
  //          we must collate results which map to the same pixel
  //          into a single term and then square it
  //          it is not correct to square and then sum individually
  
  PathToImage mapToImage;

  Spectrum d;
  // shorthand
  const PathSampler::Result &r = results[0];

  // map the result to a location in the image
  gpcpu::float2 pixel;
  mapToImage(r, x, xPath, pixel[0], pixel[1]);

  // each deposit takes the following form:
  d = (w * r.mWeight / r.mPdf) * r.mThroughput;

  // square
  d *= d;

  // handoff to deposit() 
  deposit(pixel[0], pixel[1], d);
} // end RenderFilm::record()

void RenderFilm
  ::recordVariance(const float w,
                   const PathSampler::HyperPoint &x,
                   const Path &xPath,
                   const std::vector<PathSampler::Result> &results,
                   const RenderFilm &meanImage,
                   const size_t i)
{
  // XXX TODO implement this correctly for Paths which may generate
  //          several results
  //          we must collate results which map to the same pixel
  //          into a single term and then update variance at once
  //          it is not correct otherwise
  
  PathToImage mapToImage;

  // shorthand
  const PathSampler::Result &r = results[0];

  // map the result to a location in the image
  gpcpu::float2 pixel;
  mapToImage(r, x, xPath, pixel[0], pixel[1]);

  // look up the current mean
  float Mi = meanImage.pixel(pixel[0], pixel[1]).mean();
  float spp = static_cast<float>(i) / (meanImage.getWidth() * meanImage.getHeight());
  Mi /= spp;

  // look up last i
  float iMinusOne = this->pixel(pixel[0], pixel[1])[2];

  // look up last estimate of T
  float tIMinusOne = this->pixel(pixel[0], pixel[1])[0];

  // compute X_i
  float Xi = ((w * r.mWeight / r.mPdf) * r.mThroughput).mean();

  // T_i = T_{i-1} + (i-1)/i * (X_i - M_i)^2
  float diff = Xi - Mi;
  //float Ti = tIMinusOne + (iMinusOne/i) * diff * diff;
  float Ti = tIMinusOne + (iMinusOne/(iMinusOne + 1.0f)) * diff * diff;

  // update
  //this->pixel(pixel[0], pixel[1]) = Spectrum(Ti, 0, i);
  this->pixel(pixel[0], pixel[1]) = Spectrum(Ti, 0, iMinusOne + 1.0f);
} // end RenderFilm::recordVariance()

