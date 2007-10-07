/*! \file WhittedSampler.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a PathSampler
 *         implementing Whitted-style path tracing.
 */

#ifndef WHITTED_SAMPLER_H
#define WHITTED_SAMPLER_H

#include "ArvoKirkSampler.h"

class WhittedSampler
  : public ArvoKirkSampler
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ArvoKirkSampler Parent;

    /*! Constructor accepts a maximum length for the eye subpath.
     *  \param maxEyeLength Sets Parent::mMaxEyeLength.
     */
    WhittedSampler(const unsigned int maxEyeLength = Path::static_size - 1);
}; // end WhittedSampler

#endif // WHITTED_SAMPLER_H

