/*! \file CudaRandomSequence.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to
 *         a RandomSequence class which
 *         can generate random numbers on
 *         a CUDA device.
 */

#pragma once

#include <cudamt/CudaMersenneTwister.h>
#include "../../numeric/RandomSequence.h"

class CudaRandomSequence
  : public RandomSequence
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef RandomSequence Parent;

    /*! Constructor seeds this CudaRandomSequence.
     *  \param seed The seed of the sequence.
     */
    CudaRandomSequence(const unsigned int seed = 13u);

    /*! This method fills a device vector with
     *  uniform pseudo-random numbers in the
     *  unit interval [0,1).
     *  \param v This vector will be filled with uniform random numbers.
     *  \param n The length of n.
     */
    virtual void operator()(const stdcuda::device_ptr<float> &v,
                            const size_t n);

  protected:
    /*! This object does the heavy lifting.
     */
    CudaMersenneTwister mTwister;
}; // end CudaRandomSequence

