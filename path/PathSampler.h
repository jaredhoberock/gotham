/*! \file PathSampler.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an abstract
 *         class for picking paths from path space.
 */

#ifndef PATH_SAMPLER_H
#define PATH_SAMPLER_H

#include <vector>
#include <spectrum/Spectrum.h>
#include <boost/array.hpp>
#include "Path.h"
#include "../numeric/RandomSequence.h"

class PathSampler
{
  public:
    struct Result
    {
      float mPdf;
      float mWeight;
      size_t mEyeLength;
      size_t mLightLength;
      Spectrum mThroughput;
    }; // end Result

    typedef std::vector<Result> ResultList;

    /*! \typedef Hyperpoint
     *  \brief Shorthand.
     */
    typedef boost::array<boost::array<float,5>, Path::static_size> HyperPoint;

    /*! Null destructor does nothing.
     */
    inline virtual ~PathSampler(void){;};

    /*! This virtual method constructs a Path given
     *  a HyperPoint uniquely specifying a Path in a Scene
     *  of interest.
     *  \param scene The Scene containing the environment to
     *               construct a Path in.
     *  \param x A HyperPoint uniquely specifying the Path to
     *           construct.
     *  \param p The constructed Path will be returned here.
     *  \return true if a Path could be constructed; false, otherwise.
     *  \note This method must be implemented in a derived class.
     */
    virtual bool constructPath(const Scene &scene,
                               const HyperPoint &x,
                               Path &p) = 0;

    /*! This method evaluates this Path's Monte Carlo contribution.
     *  \param scene The Scene containing the environment to construct
     *               a Path in.
     *  \param p The Path of interest assumed to be constructed by this
     *           SimpleBidirectionalSampler.
     *  \param results A list of Monte Carlo contributions, binned by
     *                 subpath length, is returned here.
     *  \note This method must be implemented in a derived class.
     */
    virtual void evaluate(const Scene &scene,
                          const Path &p,
                          std::vector<Result> &results) const = 0;

    /*! This method creates a new HyperPoint given an iterator to a sequence
     *  of real numbers each in the unit interval, [0,1).
     *  \param s A sequence of pseudo-random numbers.
     *  \param x The HyperPoint specified by the sequence will be returned here.
     */
    static void constructHyperPoint(RandomSequence &s, HyperPoint &x);
}; // end PathSampler

#endif // PATH_SAMPLER_H

