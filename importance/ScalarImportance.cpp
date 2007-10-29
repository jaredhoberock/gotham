/*! \file ScalarImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ScalarImportance class.
 */

#include "ScalarImportance.h"
#include <aliastable/AliasTable.h>
#include "../shading/FunctionAllocator.h"
using namespace boost;
class MetropolisRenderer;

void ScalarImportance
  ::preprocess(const shared_ptr<RandomSequence> &r,
               const shared_ptr<const Scene> &scene,
               const shared_ptr<PathMutator> &mutator,
               MetropolisRenderer &renderer)
{
  // by default, just set these to during the integration
  mNormalizationConstant = 1.0f;
  mInvNormalizationConstant = 1.0f;

  // now integrate
  mNormalizationConstant = estimateNormalizationConstant(*r.get(), scene, mutator, 10000);
  mInvNormalizationConstant = 1.0f / mNormalizationConstant;
} // end ScalarImportance::preprocess()

float ScalarImportance
  ::estimateNormalizationConstant(const boost::shared_ptr<RandomSequence> &r,
                                  const boost::shared_ptr<const Scene> &scene,
                                  const boost::shared_ptr<PathMutator> &mutator,
                                  const size_t n,
                                  FunctionAllocator &allocator,
                                  PathSampler::HyperPoint &x,
                                  Path &xPath)
{
  float result = 0;
  Spectrum L;

  // estimate b
  typedef std::vector<PathSampler::HyperPoint> SeedList;
  typedef std::vector<PathSampler::Result> ResultList;
  ResultList resultList;
  SeedList seeds;
  std::vector<float> seedImportance;
  float I;

  // XXX fix this
  PathSampler *sampler = const_cast<PathSampler*>(mutator->getSampler());
  for(size_t i = 0; i < n; ++i)
  {
    PathSampler::constructHyperPoint(*r, x);

    // create a Path
    if(sampler->constructPath(*scene, x, xPath))
    {
      // evaluate the Path
      resultList.clear();
      L = mutator->evaluate(xPath, resultList);

      I = evaluate(x, xPath, resultList);
      result += I;

      seeds.push_back(x);
      seedImportance.push_back(I);
    } // end if

    // free all integrands allocated in this sample
    ScatteringDistributionFunction::mPool.freeAll();
  } // end for i

  // pick a seed
  AliasTable<PathSampler::HyperPoint> aliasTable;
  aliasTable.build(seeds.begin(), seeds.end(),
                   seedImportance.begin(), seedImportance.end());
  x = aliasTable((*r)());
  Path temp;
  sampler->constructPath(*scene, x, temp);

  // copy temp to x
  temp.clone(xPath, allocator);

  // free all integrands that were allocated in the estimate
  ScatteringDistributionFunction::mPool.freeAll();
  
  return result / n;
} // end ScalarImportance::estimateNormalizationConstant()

float ScalarImportance
  ::estimateNormalizationConstant(RandomSequence &r,
                                  const boost::shared_ptr<const Scene> &scene,
                                  const boost::shared_ptr<PathMutator> &mutator,
                                  const size_t n)
{
  float result = 0;
  Spectrum L;

  // estimate b
  float I;

  // XXX remove this const_cast
  PathSampler *sampler = const_cast<PathSampler*>(mutator->getSampler());
  PathSampler::HyperPoint x;
  Path xPath;
  std::vector<PathSampler::Result> resultList;
  for(size_t i = 0; i < n; ++i)
  {
    PathSampler::constructHyperPoint(r, x);

    // create a Path
    if(sampler->constructPath(*scene, x, xPath))
    {
      // evaluate the Path
      resultList.clear();
      L = mutator->evaluate(xPath, resultList);

      I = evaluate(x, xPath, resultList);
      result += I;
    } // end if

    // free all integrands allocated in this sample
    ScatteringDistributionFunction::mPool.freeAll();
  } // end for i

  return result / n;
} // end ScalarImportance::estimateNormalizationConstant()

float ScalarImportance
  ::operator()(const PathSampler::HyperPoint &x,
               const Path &xPath,
               const std::vector<PathSampler::Result> &results)
{
  return evaluate(x,xPath,results);
} // end ScalarImportance::operator()()

