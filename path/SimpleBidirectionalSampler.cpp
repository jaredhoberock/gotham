/*! \file SimpleBidirectionalSampler.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SimpleBidirectionalSampler class.
 */

#include "SimpleBidirectionalSampler.h"
#include "../geometry/Ray.h"
#include "../shading/Material.h"
#include "../primitives/SurfacePrimitive.h"
#include "../primitives/SurfacePrimitiveList.h"
#include "../primitives/Scene.h"
#include "../shading/ScatteringDistributionFunction.h"

SimpleBidirectionalSampler
  ::SimpleBidirectionalSampler(const unsigned int maxLength)
{
  init(maxLength);
} // end SimpleBidirectionalSampler::SimpleBidirectionalSampler()

void SimpleBidirectionalSampler
  ::init(const unsigned int maxLength)
{
  mMaxPathLength = std::max<unsigned int>(3, maxLength);

  // set defaults

  //mMinEyeSubpathLength = 2;
  mMinEyeSubpathLength = 1;
  // XXX debug this
  //mMinEyeSubpathLength = 0;
  //mMinLightSubpathLength = 0;
  mMinLightSubpathLength = 1;

  // set limits on the total path
  mMinPathLength = mMinEyeSubpathLength + mMinLightSubpathLength;

  mMaxEyeSubpathLength = mMaxPathLength - mMinLightSubpathLength;
  mMaxLightSubpathLength = mMaxPathLength - mMinEyeSubpathLength;

  // construct pdfs
  constructPdfs();
} // end SimpleBidirectionalSampler::init()

SimpleBidirectionalSampler
  ::SimpleBidirectionalSampler(void)
{
  // set defaults

  //mMinEyeSubpathLength = 2;
  mMinEyeSubpathLength = 1;
  // XXX debug this
  //mMinEyeSubpathLength = 0;
  //mMinLightSubpathLength = 0;
  mMinLightSubpathLength = 1;

  // set limits on the total path
  mMaxPathLength = Path::static_size;
  mMinPathLength = mMinEyeSubpathLength + mMinLightSubpathLength;

  mMaxEyeSubpathLength = mMaxPathLength - mMinLightSubpathLength;
  mMaxLightSubpathLength = mMaxPathLength - mMinEyeSubpathLength;

  // construct pdfs
  constructPdfs();
} // end SimpleBidirectionalSampler::SimpleBidirectionalSampler()

void SimpleBidirectionalSampler
  ::constructPdfs(void)
{
  // construct mPathLengthPdf
  std::vector<size_t> lengths;
  for(size_t pathLength = mMinPathLength;
      pathLength <= mMaxPathLength;
      ++pathLength)
    lengths.push_back(pathLength);

  // construct a constant pdf over path lengths
  std::vector<float> probabilities(lengths.size());
  std::fill(probabilities.begin(), probabilities.end(), 1.0f);

  // build the pdf
  mPathLengthPdf.build(lengths.begin(), lengths.end(),
                       probabilities.begin(), probabilities.end());

  // build the subpath length pdfs
  mSubpathLengthPdfs.resize(mMaxPathLength - mMinPathLength + 1);
  for(size_t pathLength = mMinPathLength;
      pathLength <= mMaxPathLength;
      ++pathLength)
  {
    std::vector<std::pair<size_t, size_t> > subpathLengths;
    for(size_t eyeLength = mMinEyeSubpathLength;
        eyeLength <= pathLength - mMinLightSubpathLength && eyeLength <= mMaxEyeSubpathLength;
        ++eyeLength)
    {
      size_t lightLength = pathLength - eyeLength;
      if(lightLength >= mMinLightSubpathLength
         && lightLength <= mMaxLightSubpathLength)
      {
        subpathLengths.push_back(std::make_pair(eyeLength, lightLength));
      } // end if
    } // end for eyeLength

    // construct a constant pdf over subpath lengths
    std::vector<float> probabilities(subpathLengths.size());
    std::fill(probabilities.begin(), probabilities.end(), 1.0f);

    mSubpathLengthPdfs[pathLength - mMinPathLength].build(subpathLengths.begin(), subpathLengths.end(),
                                                          probabilities.begin(), probabilities.end());
  } // end for i
} // end SimpleBidirectionalSampler::constructPdfs()

bool SimpleBidirectionalSampler
  ::constructPath(const Scene &scene,
                  const HyperPoint &x,
                  Path &p)
{
  // we use the 4th coordinates of the first two points in the
  // HyperPoint to choose the path lengths
  // because they are not used in either of the surface area
  // sampling functions

  // choose the length of the full path
  float lengthPdf = 0.0f;
  size_t numVertices = mPathLengthPdf(x[0][4], lengthPdf);

  // choose the lengths of the subpaths
  float subpathLengthPdf = 0.0;
  std::pair<size_t, size_t> subpathLengths
    = mSubpathLengthPdfs[numVertices - mMinPathLength](x[1][4], subpathLengthPdf);

  // construct the paths
  p.clear();
  if(!constructEyePath(scene, x, subpathLengths.first, p)) return false;
  if(!constructLightPath(scene, x, subpathLengths.second, p)) return false;

  // tack length pdfs onto the end of the eye subpath (if it exists)
  // else, put it on the end of the light path (which happens to be the first vertex)
  if(p.getSubpathLengths()[0] > 0)
  {
    p[p.getSubpathLengths()[0] - 1].mAccumulatedPdf *= lengthPdf * subpathLengthPdf;
    p[p.getSubpathLengths()[0] - 1].mPdf *= lengthPdf * subpathLengthPdf;
  } // end if
  else
  {
    p[0].mAccumulatedPdf *= lengthPdf * subpathLengthPdf;
    p[0].mPdf *= lengthPdf * subpathLengthPdf;
  } // end else

  // do we need to connect the subpaths?
  if(p.getSubpathLengths()[0] &&
     p.getSubpathLengths()[1])
  {
    p.connect(p[p.getSubpathLengths()[0]-1], p[p.getSubpathLengths()[0]]);
  } // end if

  return true;
} // end SimpleBidirectionalSampler::constructPath()

bool SimpleBidirectionalSampler
  ::constructEyePath(const Scene &scene,
                     const HyperPoint &x,
                     const size_t numVertices,
                     Path &p) const
{
  // treat every other coordinate of x as an eye coordinate,
  // and vice versa for the light path
  unsigned int justAdded = Path::INSERT_FAILED;
  for(unsigned int i = 0;
      i != numVertices;
      ++i)
  {
    if(i == 0)
    {
      // we define the pixel location to come as the first coordinate
      // so use the next coordinate to pick an aperture point
      justAdded = p.insert(0, &scene, scene.getSensors(), false,
                           x[1][0], x[1][1], x[1][2], x[1][3]);
    } // end if
    else if(i == 1)
    {
      // we define the pixel location to come as the first coordinate,
      // so use the 0th coordinate to pick a pixel location
      justAdded = p.insert(justAdded, &scene, true, false,
                           x[0][0], x[0][1], x[0][2]);
    } // end else if
    else
    {
      justAdded = p.insert(justAdded, &scene, true, true,
                           x[i][0], x[i][1], x[i][2]);
    } // end else

    // stop?
    if(justAdded == Path::INSERT_FAILED)
    {
      return false;
    } // end if
  } // end for i,j

  return true;
} // end SimpleBidirectionalSampler::constructEyePath()

bool SimpleBidirectionalSampler
  ::constructLightPath(const Scene &scene,
                       const HyperPoint &x,
                       const size_t numVertices,
                       Path &p) const
{
  // treat every other coordinate of x as an eye coordinate,
  // and vice versa for the light path
  
  // leave enough room for #numVertices light vertices after the eye vertices
  unsigned int justAdded = p.getSubpathLengths()[0] + numVertices - 1;
  unsigned int j = 0;
  for(unsigned int i = p.getSubpathLengths()[0];
      j != numVertices;
      i += 1, ++j)
  {
    if(j == 0)
    {
      justAdded = p.insert(justAdded, &scene, scene.getEmitters(), true,
                           x[i][0], x[i][1], x[i][2], x[i][3]);
    } // end if
    else if(j == 1)
    {
      justAdded = p.insert(justAdded, &scene, false, false,
                           x[i][0], x[i][1], x[i][2]);
    } // end else if
    else
    {
      justAdded = p.insert(justAdded, &scene, false, true,
                           x[i][0], x[i][1], x[i][2]);
    } // end else

    // stop?
    if(justAdded == Path::INSERT_FAILED)
    {
      return false;
    } // end if
  } // end for i,j

  return true;
} // end SimpleBidirectionalSampler::constructLightPath()

void SimpleBidirectionalSampler
  ::evaluate(const Scene &scene,
             const Path &p,
             std::vector<Result> &results) const
{
  size_t eyeLength = p.getSubpathLengths()[0];
  size_t lightLength = p.getSubpathLengths()[1];

  assert(eyeLength != 0);

  // connect the eye and light subpaths
  if(eyeLength != 0
     && lightLength != 0)
  {
    const PathVertex &e = p[eyeLength-1];
    const PathVertex &l = p[eyeLength];

    if(!e.mScattering->isSpecular() && !l.mScattering->isSpecular())
    {
      Spectrum L = e.mThroughput * l.mThroughput;
      L *= p.computeThroughputAssumeVisibility(eyeLength, e,
                                               lightLength, l);

      if(!L.isBlack()
         && !scene.intersect(Ray(e.mDg.getPoint(), l.mDg.getPoint())))
      {
        // add a new result
        results.resize(results.size() + 1);
        Result &r = results.back();

        // multiply by the connection throughput
        r.mThroughput = L;

        // set pdf, weight, and (s,t)
        r.mPdf = e.mAccumulatedPdf * l.mAccumulatedPdf;
        r.mWeight = computeMultipleImportanceSamplingWeight(scene, p);
        r.mEyeLength = eyeLength;
        r.mLightLength = lightLength;
      } // end if
    } // end if
  } // end if
  //else if(lightLength == 0)
  //{
  //  // evaluate emission
  //  const PathVertex &e = p[eyeLength-1];

  //  const Material *m = e.mDg.getSurface()->getMaterial();
  //  Spectrum L = m->evaluateEmission(e.mDg)->evaluate(e.mToPrev, e.mDg);
  //  if(L != Spectrum::black())
  //  {
  //    L *= e.mThroughput;

  //    // add a new result
  //    results.resize(results.size() + 1);
  //    Result &r = results.back();
  //    r.mThroughput = L;
  //    r.mPdf = e.mAccumulatedPdf;
  //    r.mWeight = computeMultipleImportanceSamplingWeight(scene, p);
  //    r.mEyeLength = eyeLength;
  //    r.mLightLength = 0;
  //  } // end if
  //} // end else if
  //else
  //{
  //  // evaluate sensing
  //  const PathVertex &l = p[0];

  //  const Material *m = l.mDg.getSurface()->getMaterial();
  //  Spectrum L = m->evaluateSensor(l.mDg)->evaluate(l.mToNext, l.mDg);
  //  if(L != Spectrum::black())
  //  {
  //    L *= l.mThroughput;

  //    // add a new result
  //    results.resize(results.size() + 1);
  //    Result &r = results.back();
  //    r.mThroughput = L;
  //    r.mPdf = l.mAccumulatedPdf;
  //    r.mWeight = computeMultipleImportanceSamplingWeight(scene, p);
  //    r.mEyeLength = 0;
  //    r.mLightLength = lightLength;
  //  } // end if
  //} // end else
} // end SimpleBidirectionalSampler::evaluate()

float SimpleBidirectionalSampler
  ::computeAreaProductMeasurePdf(const Scene &scene,
                                 const Path &p,
                                 const size_t s,
                                 const size_t t) const
{
  // start with the pdf of choosing this path length
  float pdf = mPathLengthPdf.evaluatePdf(s+t);

  // multiply by the pdf of choosing the subpath lengths
  pdf *= mSubpathLengthPdfs[s+t-mMinPathLength].evaluatePdf(std::make_pair(t,s));

  // compute the pdf of the eye subpath
  for(size_t i = 0;
      i != t;
      ++i)
  {
    float term = 0;
    if(i == 0)
    {
      term = scene.getSensors()->evaluateSurfaceAreaPdf(p[i].mSurface, p[i].mDg);
    } // end if
    else
    {
      // compute the solid angle pdf
      if(i == 1)
      {
        term = p[i-1].mSensor->evaluatePdf(p[i-1].mToNext,
                                           p[i-1].mDg);
      } // end for i
      else
      {
        if(p[i-1].mScattering->isSpecular())
        {
          term = 1.0f;
        } // end if
        else
        {
          term = p[i-1].mScattering->evaluatePdf(p[i-1].mToPrev,
                                                 p[i-1].mDg,
                                                 p[i-1].mToNext);
        } // end else
      } // end else

      pdf *= term;

      // convert to projected solid angle pdf
      pdf /= p[i-1].mDg.getNormal().absDot(p[i-1].mToNext);

      // convert to area product pdf
      pdf *= p[i-1].mNextGeometricTerm;
    } // end else
  } // end for i

  //assert(pdf == pdf);

  // compute the pdf of the light subpath

  // XXX compute the area product pdf of choosing the first vertex

  // compute the pdf of the light subpath
  size_t j = 0;
  for(size_t i = t + s - 1;
      j != s;
      --i, ++j)
  {
    if(j == 0)
    {
      pdf *= scene.getEmitters()->evaluateSurfaceAreaPdf(p[i].mSurface, p[i].mDg);
    } // end if
    else
    {
      // compute the solid angle pdf
      if(j == 1)
      {
        pdf *= p[i+1].mEmission->evaluatePdf(p[i+1].mToPrev, p[i+1].mDg);
      } // end if
      else
      {
        if(p[i+1].mScattering->isSpecular())
        {
          pdf *= 1.0f;
        } // end if
        else
        {
          pdf *= p[i+1].mScattering->evaluatePdf(p[i+1].mToNext,
                                                 p[i+1].mDg,
                                                 p[i+1].mToPrev);
        } // end else
      } // end else

      // convert to projected solid angle pdf
      pdf /= p[i+1].mDg.getNormal().absDot(p[i+1].mToPrev);

      // convert to area product pdf
      pdf *= p[i+1].mPreviousGeometricTerm;
    } // end else
  } // end for i

  return pdf;
} // end SimpleBidirectionalSampler::computeAreaProductMeasurePdf()

float SimpleBidirectionalSampler
  ::computeMultipleImportanceSamplingWeight(const Scene &scene, const Path &p) const
{
  float numerator = computeAreaProductMeasurePdf(scene, p,
                                                 p.getSubpathLengths()[1],
                                                 p.getSubpathLengths()[0]);

  // power heuristic
  numerator *= numerator;

  float denominator = 0;

  size_t k = p.getSubpathLengths().sum();
  for(size_t t = mMinEyeSubpathLength;
      t <= k - mMinLightSubpathLength;
      ++t)
  {
    size_t s = k - t;

    if(t <= mMaxEyeSubpathLength &&
       s <= mMaxLightSubpathLength)
    {
      float term = computeAreaProductMeasurePdf(scene, p, s, t);

      // power heuristic
      denominator += term*term;
    } // end if
  } // end for t

  if(denominator == 0) std::cerr << "zero denominator" << std::endl;

  float result = numerator / denominator;

  return result;
} // end SimpleBidirectionalSampler::computeMultipleImportanceSamplingWeight()

