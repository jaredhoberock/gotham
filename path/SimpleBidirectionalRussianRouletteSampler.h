/*! \file SimpleBidirectionalRussianRouletteSampler.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a PathSampler
 *         performing Russian roulette at each bounce
 *         and which only computes one connection per
 *         pair of subpaths.
 */

#ifndef SIMPLE_BIDIRECTIONAL_RUSSIAN_ROULETTE_SAMPLER_H
#define SIMPLE_BIDIRECTIONAL_RUSSIAN_ROULETTE_SAMPLER_H

#include "PathSampler.h"
#include "RussianRoulette.h"

class SimpleBidirectionalRussianRouletteSampler
  : public PathSampler
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef PathSampler Parent;

    /*! Constructor accepts a maximum path length.
     *  \param roulette A shared pointer to a RussianRoulette function.
     *  \param maxLength Sets the maximum length for paths
     *                   created by this SimpleBidirectionalSampler.
     */
    SimpleBidirectionalRussianRouletteSampler(const boost::shared_ptr<RussianRoulette> &roulette,
                                              const size_t maxLength);

    /*! This method constructs a Path given a
     *  HyperPoint uniquely specifying a Path in a
     *  Scene of interest.
     *  \param scene The Scene containing the environment to
     *               construct a Path in.
     *  \param context A ShadingContext for evaluating shaders.
     *  \param x A HyperPoint uniquely specifying the Path to
     *           construct.
     *  \param p The constructed Path will be returned here.
     *  \return true if a Path could be constructed; false, otherwise.
     */
    virtual bool constructPath(const Scene &scene,
                               ShadingContext &context,
                               const HyperPoint &x,
                               Path &p);

    /*! This method evaluates this Path's Monte Carlo contribution.
     *  \param scene The Scene containing the environment to construct
     *               a Path in.
     *  \param p The Path of interest assumed to be constructed by this
     *           SimpleBidirectionalSampler.
     *  \param results A list of Monte Carlo contributions, binned by
     *                 subpath length, is returned here.
     */
    virtual void evaluate(const Scene &scene,
                          const Path &p,
                          std::vector<Result> &results) const;

  protected:
    unsigned int mMaxPathLength;

    boost::shared_ptr<RussianRoulette> mRoulette;
}; // end SimpleBidirectionalRussianRouletteSampler

#endif // SIMPLE_BIDIRECTIONAL_RUSSIAN_ROULETTE_SAMPLER_H

