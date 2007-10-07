/*! \file KelemenSampler.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a PathSampler
 *         implementing the bidirectional path sampling
 *         technique described by Kelemen et al 02.
 */

#ifndef KELEMEN_SAMPLER_H
#define KELEMEN_SAMPLER_H

#include "PathSampler.h"
class RussianRoulette;

class KelemenSampler
  : public PathSampler
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef PathSampler Parent;

    /*! Constructor accepts a maximum path length.
     *  \param roulette A shared pointer to a Russian roulette function.
     *  \param maxLength Sets the maximum length for
     *                   paths created by this KelemenSampler.
     */
    KelemenSampler(const boost::shared_ptr<RussianRoulette> &roulette,
                   const unsigned int maxLength);

    /*! This method constructs a Path given a HyperPoint uniquely
     *  specifying a Path in a Scene of interest.
     *  \param scene The Scene containing the environment to
     *               construct a Path in.
     *  \param x A HyperPoint uniquely specifying the Path to
     *           construct.
     *  \param p The constructed Path will be returned here.
     *  \return true if a Path could be constructed; false, otherwise.
     */
    virtual bool constructPath(const Scene &scene,
                               const HyperPoint &x,
                               Path &p);

    virtual bool constructPathInterleaved(const Scene &scene,
                                          const HyperPoint &x,
                                          Path &p) const;

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
    /*! This method constructs the eye subpath of a given Path.
     *  \param scene The Scene containing the environment to
     *               construct a Path in.
     *  \param x A HyperPoint uniquely specifying the Path to
     *           construct.
     *  \param p This Path's eye subpath will be updated to
     *           reflect a new subpath.
     *  \return true if an eye subpath could be constructed;
     *          false, otherwise.
     */
    virtual bool constructEyePath(const Scene &scene,
                                  const HyperPoint &x,
                                  Path &p) const;

    /*! This method constructs the light subpath of a given Path.
     *  \param scene The Scene containing the environment to
     *               construct a Path in.
     *  \param x A HyperPoint uniquely specifying the Path to
     *           construct.
     *  \param p This Path's light subpath will be updated to
     *           reflect a new subpath.
     *  \return true if an light subpath could be constructed;
     *          false, otherwise.
     */
    virtual bool constructLightPath(const Scene &scene,
                                    const HyperPoint &x,
                                    Path &p) const;

    float computeWeight(const Scene &scene,
                        const Path::const_iterator &lLast,
                        const size_t s,
                        const size_t lightSubpathLength,
                        const Path::const_iterator &eLast,
                        const size_t t,
                        const size_t eyeSubpathLength,
                        const Vector &connection,
                        const float g,
                        const RussianRoulette &roulette) const;

    unsigned int mMaxPathLength;

    /*! A RussianRoulette function.
     */
    boost::shared_ptr<RussianRoulette> mRoulette;
}; // end KelemenSampler

#endif // KELEMEN_SAMPLER_H

