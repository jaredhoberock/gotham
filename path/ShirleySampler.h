/*! \file ShirleySampler.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a PathSampler
 *         which carefully computes "direct" illumination.
 */

#ifndef SHRILEY_SAMPLER_H
#define SHRILEY_SAMPLER_H

#include "WhittedSampler.h"

class ShirleySampler
  : public WhittedSampler
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef WhittedSampler Parent;

    /*! Constructor accepts a maximum length for eye paths.
     *  \param maxEyeLength Sets Parent::mMaxEyeLength.
     */
    ShirleySampler(const unsigned int maxEyeLength = Path::static_size - 1);

    virtual void evaluate(const Scene &scene,
                          const Path &p,
                          std::vector<Result> &results) const;
}; // end ShirleySampler

#endif // SHRILEY_SAMPLER_H

