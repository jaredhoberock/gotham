/*! \file PhotonRecord.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PhotonRecord class.
 */

#include "PhotonRecord.h"
#include "../primitives/SurfacePrimitive.h"
#include <fstream>

PhotonRecord
  ::PhotonRecord(void)
    :Parent0(),Parent1(),mFilename("")
{
  ;
} // end PhotonRecord::PhotonRecord()

PhotonRecord
  ::PhotonRecord(const size_t n,
                 const std::string &filename)
    :Parent0(),Parent1(),mFilename(filename)
{
  reset();
  reserve(n);
} // end PhotonRecord::PhotonRecord()

void PhotonRecord
  ::reset(void)
{
  mNumDeposits = 0;
} // end RenderFilm::reset()

size_t PhotonRecord
  ::getNumDeposits(void) const
{
  return mNumDeposits;
} // end PhotonRecord::getNumDeposits()

void PhotonRecord
  ::postprocess(void)
{
  // postprocess the parent first
  Parent0::postprocess();

  if(mFilename != "")
  {
    std::fstream outfile(mFilename.c_str(), std::fstream::out);
    if(outfile.is_open())
    {
      outfile << *this << std::endl;

      std::cout << "Wrote photons to: " << getFilename() << std::endl;
    } // end if
    else
    {
      std::cout << "Error: Unable to write photons to: " << getFilename() << std::endl;
    } // end else
  } // end if
} // end PhotonRecord::postprocess()

const std::string &PhotonRecord
  ::getFilename(void) const
{
  return mFilename;
} // end PhotonRecord::getFilename()

void PhotonRecord
  ::setFilename(const std::string &filename)
{
  mFilename = filename;
} // end PhotonRecord::setFilename()

void PhotonRecord
  ::record(const float w,
           const Path &xPath,
           const PathSampler::Result &r,
           PhotonList &photons) const
{
  // this is made int because we subtract from it
  int totalPathLength = xPath.getSubpathLengths().sum();

  // deposit photons beginning at the second eye vertex
  // all the way up to the penultimate eye vertex
  PhotonList::iterator photon = photons.begin() + 1;
  //for(Path::const_iterator v = xPath.begin() + 1;
  //    v < xPath.begin() + r.mEyeLength - 1;
  //    ++v, ++photon)
  //{
  //  // for eye vertices, each photon's power
  //  // should be equal to r's throughput divided by
  //  // v's accumulated throughput
  //  // the pdf should work the same way
  //  float pdf = r.mPdf / v->mAccumulatedPdf;
  //  Spectrum throughput = r.mThroughput / v->mThroughput;

  //  // accumulate power
  //  //photon->mPower += (w * r.mWeight / pdf) * throughput;
  //  photon->mPower += (w / pdf) * throughput;
  //} // end for i

  // accumulate power for each vertex on the light subpath
  // deposit photons beginning at the "last" light vertex
  // all the way back to one vertex before the light
  photon = photons.begin() + totalPathLength - r.mLightLength;
  for(Path::const_iterator v = xPath.begin() + totalPathLength - r.mLightLength;
      v < xPath.begin() + totalPathLength - 1;
      ++v, ++photon)
  {
    // for light vertices, each photon's power
    // is simply equal to v's accumulated throughput
    // the pdf should work the same way
    float pdf = v->mAccumulatedPdf;
    Spectrum throughput = v->mThroughput;

    // accumulate power
    //photon->mPower += (w * r.mWeight / pdf) * throughput;
    photon->mPower += (w / pdf) * throughput;

    std::cerr << "deposit light vertex: " << throughput / pdf << std::endl;
  } // end for i
} // end PhotonRecord::record()

void PhotonRecord
  ::record(const float w,
           const PathSampler::HyperPoint &x,
           const Path &xPath,
           const std::vector<PathSampler::Result> &results)
{
  Spectrum power;
  Point p;
  Vector wi;

  // this is made int because we subtract from it
  int totalPathLength = xPath.getSubpathLengths().sum();

  // create a Photon slot for every PathVertex
  typedef boost::array<Photon,Path::static_size> Photons;
  Photons photons;
  Path::const_iterator v = xPath.begin();
  for(Photons::iterator p = photons.begin();
      p != photons.begin() + totalPathLength;
      ++p, ++v)
  {
    // initialize the Photon
    p->mPoint = v->mDg.getPoint();
    p->mWi = v->mToNext;
    p->mPower = Spectrum::black();
  } // end for p

  // accumulate photon power for each vertex on the eye subpath
  for(std::vector<PathSampler::Result>::const_iterator r = results.begin();
      r != results.end();
      ++r)
  {
    // record all bounces except for the shadow edge here 
    //record(w, xPath, *r, photons);

    // handle the shadow ray path edge here
    // XXX this case adds complexity
    //     it might be unbiased if we ignored this case completely
    // handle the last eye vertex specially
    // the wi is always different, so create a new photon rather
    // than accumulate to one in our list
    // don't deposit if we hit a light source with the eye path
    //if(r->mEyeLength != 0 && r->mLightLength != 0)
    if(r->mEyeLength == 2 && r->mLightLength == 1)
    {
      const PathVertex &e = xPath[r->mEyeLength-1];
      const PathVertex &l = xPath[xPath.getSubpathLengths().sum() - r->mLightLength];

      Vector wi = l.mDg.getPoint() - e.mDg.getPoint();
      float d2 = wi.dot(wi);
      float d = sqrtf(d2);
      wi /= d;
      Spectrum f(1,1,1);

      // evaluate bsdf on light end
      if(r->mLightLength == 1)
      {
        f *= l.mEmission->evaluate(-wi, l.mDg);
      } // end if
      else
      {
        f *= l.mScattering->evaluate(l.mToNext, l.mDg, -wi);
      } // end else

      // modulate by geometry term
      float g = Path::computeG(e.mDg.getNormal(), wi, l.mDg.getNormal(), d2);
      f *= g;
      f /= e.mDg.getNormal().absDot(wi);


      float pdf = e.mAccumulatedPdf * l.mAccumulatedPdf;
      //float pdf = l.mAccumulatedPdf / e.mAccumulatedPdf;

      // compute throughput
      Spectrum throughput = f * l.mThroughput;

      // compute power
      //power = (w * r->mWeight / pdf) * throughput;
      power = throughput / pdf;

      // hand off to deposit()
      deposit(e.mDg.getPoint(), wi, power);

      //std::cerr << "deposit: " << throughput / pdf << std::endl;
    } // end if
  } // end for r

  // now deposit each photon
  for(Photons::const_iterator p = photons.begin();
      p != photons.begin() + totalPathLength;
      ++p)
  {
    // don't deposit zero power photons
    if(!p->mPower.isBlack())
    {
      deposit(p->mPoint, p->mWi, p->mPower);
    } // end if
  } // end for p
} // end RenderFilm::record()

void PhotonRecord
  ::deposit(const Point &x,
            const Vector &wi,
            const Spectrum &p)
{
  if(size() < capacity())
  {
    Parent1::resize(Parent1::size() + 1);

    Parent1::back().mPoint = x;
    Parent1::back().mWi = wi;
    Parent1::back().mPower = p;
  } // end if
} // end PhotonRecord::deposit()

void PhotonRecord
  ::scale(const float s)
{
  float temp = 1.0f / size();
  size_t filtered = 0;
  for(iterator p = begin();
      p != end();
      ++p)
  {
    // kill NaNs
    // XXX this should be a separate filter method or something
    if(p->mPower[0] != p->mPower[0]
       || p->mPower[1] != p->mPower[1]
       || p->mPower[2] != p->mPower[2])
    {
      p->mPower = Spectrum::black();
      ++filtered;
    } // end if

    // XXX apparently we shouldn't scale by N samples
    //p->mPower *= s;
    p->mPower *= temp;
  } // end for p

  std::cerr << "PhotonRecord::scale(): Fix me!" << std::endl;

  std::cerr << "PhotonRecord::scale(): filtered " << 100.0f * (float)filtered / size() << "% of photons." << std::endl;
} // end PhotonRecord::scale()

