/*! \file PathMutator.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PathMutator class.
 */

#include "PathMutator.h"
using namespace boost;

PathMutator
  ::PathMutator(void)
{
  ;
} // end PathMutator::PathMutator()

PathMutator
  ::PathMutator(const shared_ptr<RandomSequence> &s,
                const shared_ptr<PathSampler> &sampler)
{
  setRandomSequence(s);
  setSampler(sampler);
} // end PathMutator::PathMutator()

PathMutator
  ::PathMutator(const shared_ptr<PathSampler> &sampler)
{
  setSampler(sampler);
} // end PathMutator::PathMutator()

void PathMutator
  ::setRandomSequence(const boost::shared_ptr<RandomSequence> &s)
{
  mRandomSequence = s;
} // end PathMutator::setRandomSequence()

int PathMutator
  ::operator()(const PathSampler::HyperPoint &x,
               const Path &a,
               PathSampler::HyperPoint &y,
               Path &b)
{
  return mutate(x,a,y,b);
} // end PathMutator::operator()()

void PathMutator
  ::setScene(const shared_ptr<const Scene> &s)
{
  mScene = s;
} // end PathMutator::setScene()

void PathMutator
  ::setSampler(const shared_ptr<PathSampler> &sampler)
{
  mSampler = sampler;
} // end PathMutator::setSampler()

const PathSampler *PathMutator
  ::getSampler(void) const
{
  return mSampler.get();
} // end PathMutator::getSampler()

void PathMutator
  ::preprocess(void)
{
  ;
} // end PathMutator::preprocess()

void PathMutator
  ::postprocess(void)
{
  ;
} // end PathMutator::postprocess()

void PathMutator
  ::setShadingContext(const boost::shared_ptr<ShadingContext> &s)
{
  mShadingContext = s;
} // end PathMutator::setShadingContext()

