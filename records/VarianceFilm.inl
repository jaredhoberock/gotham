/*! \file VarianceFilm.inl
 *  \author Jared Hoberock
 *  \brief Inline file for VarianceFilm.h.
 */

#include "VarianceFilm.h"

VarianceFilm
  ::VarianceFilm(void)
    :Parent()
{
  ;
} // end VarianceFilm::VarianceFilm()

VarianceFilm
  ::VarianceFilm(const unsigned int width,
                 const unsigned int height,
                 const boost::shared_ptr<RandomAccessFilm> &estimate,
                 const std::string &filename,
                 const std::string &varianceFilename)
    :Parent(width, height, filename),mMeanEstimate(estimate)
{
  mVariance.setFilename(varianceFilename);
} // end VarianceFilm::VarianceFilm()

void VarianceFilm
  ::preprocess(void)
{
  Parent::preprocess();
  mVariance.preprocess();
} // end VarianceFilm::preprocess()

void VarianceFilm
  ::postprocess(void)
{
  Parent::postprocess();
  mVariance.postprocess();
} // end VarianceFilm::postprocess()

void VarianceFilm
  ::record(const float w,
           const PathSampler::HyperPoint &x,
           const Path &xPath,
           const std::vector<PathSampler::Result> &results)
{
  // call the Parent first
  Parent::record(w, x, xPath, results);

  // XXX TODO make this a member of the Parent and use its member for this
  PathToImage mapToImage;

  // XXX i think this isn't right
  // Var[F(X)] = Integral (f(x) - mean)^2 dx
  //
  // Var[F(X)] ~ Sum w(xi) * (f(xi) - mean)^2
  //                  ----------------
  //                      N * p(xi)
  
  Spectrum d;
  Spectrum diff;
  for(size_t i = 0;
      i != results.size();
      ++i)
  {
    // shorthand
    const PathSampler::Result &r = results[i];

    // map the result to a location in the image
    gpcpu::float2 pixel;
    mapToImage(r, x, xPath, pixel[0], pixel[1]);

    const Spectrum &mean = mMeanEstimate->pixel(pixel[0], pixel[1]);

    //diff = r.mThroughput - mean;
    //diff *= diff;

    //// each deposit takes the following form:
    //d = (w * r.mWeight / r.mPdf) * diff;

    // this looks like it might be correct
    diff = (r.mWeight / r.mPdf) * r.mThroughput;
    diff -= mean;
    diff *= diff;

    // remember that the metropolis renderer passes the pdf
    // (importance / b) baked into w (which also includes the accept probability)
    d = w * diff;

    // handoff to deposit() 
    mVariance.deposit(pixel[0], pixel[1], d);

    const Spectrum &val = static_cast<const RenderFilm&>(mVariance).pixel(pixel[0], pixel[1]);
  } // end for r
} // end VarianceFilm::record()

void VarianceFilm
  ::fill(const Pixel &v)
{
  Parent::fill(v);
  mVariance.fill(v);
} // end VarianceFilm::fill()

void VarianceFilm
  ::scale(const Pixel &s)
{
  Parent::scale(s);
  mVariance.scale(s);
} // end VarianceFilm::scale()

void VarianceFilm
  ::resize(const unsigned int width,
           const unsigned int height)
{
  Parent::resize(width,height);
  mVariance.resize(width,height);
} // end VarianceFilm::resize()

