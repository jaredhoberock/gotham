/*! \file KajiyaSampler.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a PathSampler
 *         that performs Kajiya-style forward
 *         path tracing.
 */

#ifndef KAJIYA_SAMPLER_H
#define KAJIYA_SAMPLER_H

#include "PathSampler.h"
#include <vector>

class Path;
class Scene;

class KajiyaSampler
  : public PathSampler
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef PathSampler Parent;

    /*! Constructor accepts a maximum length for the eye subpath.
     *  \param maxEyeLength Sets mMaxEyeLength.
     */
    KajiyaSampler(const unsigned int maxEyeLength = Path::static_size - 1);

    virtual bool constructPath(const Scene &scene,
                               const HyperPoint &x,
                               Path &p);

    virtual void evaluate(const Scene &scene,
                          const Path &p,
                          std::vector<Result> &results) const;

  protected:
    unsigned int mMaxEyeLength;
}; // end KajiyaSampler

#endif // KAJIYA_SAMPLER_H

