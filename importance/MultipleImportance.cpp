/*! \file MultipleImportance.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of MultipleImportance class.
 */

#include "MultipleImportance.h"
#include "LuminanceImportance.h"
#include "ConstantImportance.h"
#include "InverseLuminanceImportance.h"
#include "EqualVisitImportance.h"
#include <bittricks/bittricks.h>

MultipleImportance
  ::~MultipleImportance(void)
{
  for(size_t i = 0; i != mStrategies.size(); ++i)
  {
    delete mStrategies[i];
  } // end for i

  mStrategies.clear();
} // end MultipleImportance::~MultipleImportance()

void MultipleImportance
  ::preprocess(const boost::shared_ptr<RandomSequence> &r,
               const boost::shared_ptr<const Scene> &scene,
               const boost::shared_ptr<PathMutator> &mutator,
               MetropolisRenderer &renderer)
{
  // grab the seqeuence
  mRandomSequence = r;

  // make some strategies
  // XXX generalize
  mStrategies.push_back(new LuminanceImportance());
  mStrategies.push_back(new ConstantImportance());
  //mStrategies.push_back(new InverseLuminanceImportance());
  //mStrategies.push_back(new EqualVisitImportance());

  // preprocess each strategy
  for(size_t i = 0; i < mStrategies.size(); ++i)
  {
    mStrategies[i]->preprocess(r,scene,mutator,renderer);
  } // end for i

  Parent::preprocess(r,scene,mutator,renderer);
} // end MultipleImportance::preprocess()

float MultipleImportance
  ::evaluate(const PathSampler::HyperPoint &x,
             const Path &xPath,
             const std::vector<PathSampler::Result> &results)
{
  // pick a function uniformly at random
  float u = (*mRandomSequence)();
  float invPdf = static_cast<float>(mStrategies.size());

  size_t i = ifloor(u * invPdf);
  ScalarImportance *strategy = mStrategies[i];

  float result = strategy->evaluate(x,xPath,results);

  // scale result by f's inverse normalization constant
  result *= strategy->getInvNormalizationConstant();

  //// divide by pdf
  //return result * invPdf;
  return result;
} // end MultipleImportance::evaluate()

