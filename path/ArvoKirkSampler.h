/*! \file ArvoKirkSampler.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a PathSampler
 *         that performs forward path tracing with
 *         Russian roulette.
 */

#ifndef ARVO_KIRK_SAMPLER_H
#define ARVO_KIRK_SAMPLER_H

#include "KajiyaSampler.h"
class RussianRoulette;
#include <vector>

class ArvoKirkSampler
  : public KajiyaSampler
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef KajiyaSampler Parent;

    /*! Constructor accepts a maximum length for the eye subpath.
     *  \param roulette A shared pointer to a RussianRoulette function.
     *  \param maxEyeLength Sets Parent::mMaxEyeLength.
     */
    ArvoKirkSampler(const boost::shared_ptr<RussianRoulette> &roulette,
                    const unsigned int maxEyeLength = Path::static_size - 1);

    /*! This method constructs a Path using forward path tracing and termination
     *  with Russian roulette.
     *  \param scene The Scene containing the Path.
     *  \param x The HyperPoint uniquely describing a new Path to create.
     *  \param p The constructed Path will be returned here.
     *  \return true, if p could be successfully constructed; false, otherwise.
     */
    virtual bool constructPath(const Scene &scene,
                               const HyperPoint &x,
                               Path &p);

  protected:
    boost::shared_ptr<RussianRoulette> mRoulette;
}; // end ArvoKirkSampler

#endif // ARVO_KIRK_SAMPLER_H

