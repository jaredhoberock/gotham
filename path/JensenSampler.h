/*! \file JensenSampler.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a PathSampler
 *         which estimates indirect illumination
 *         from PhotonMaps.
 */

#ifndef JENSEN_SAMPLER_H
#define JENSEN_SAMPLER_H

#include "ShirleySampler.h"
#include "../records/PhotonMap.h"
#include <hilbertsequence/HilbertSequence.h>

class JensenSampler
  : public ShirleySampler
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ShirleySampler Parent;

    /*! This method sets the global PhotonMap.
     *  \param pm Sets mGlobalMap
     */
    void setGlobalMap(const boost::shared_ptr<PhotonMap> &pm);

    /*! This method sets the caustic PhotonMap.
     *  \param pm Sets mCausticMap
     */
    void setCausticMap(const boost::shared_ptr<PhotonMap> &pm);

    virtual void evaluate(const Scene &scene,
                          const Path &p,
                          std::vector<Result> &results) const;

    /*! This method sets the number of final gather strata.
     *  \param x The number of strata in 'x'.
     *  \param y The number of strata in 'y'.
     */
    void setFinalGatherStrata(const size_t x, const size_t y);

    /*! This method sets the number of photons to use in each
     *  radiance estimate.
     *  \param n Sets the number of photons to use in each final
     *           gather estimate.
     */
    void setFinalGatherPhotons(const size_t n);

  protected:
    struct PhotonGatherer
      : public std::vector<std::pair<float, const Photon*> >
    {
      inline virtual ~PhotonGatherer(void){;};

      virtual void operator()(const Photon &p, const float d2, float &maxDist2);

      unsigned int mPhotonsVisited;
    }; // end PhotonGatherer

    struct PhotonKernel
    {
      inline PhotonKernel(const Point &x, const float &maxDist2);
      inline virtual ~PhotonKernel(void){;};
      virtual float operator()(const Photon &p, const float d2) const = 0;
      Point mX;
      float mMaxDist2;
      float mInvMaxDist2;
    }; // end PhotonKernel

    struct ConstantKernel
      : public PhotonKernel
    {
      typedef PhotonKernel Parent;
      inline ConstantKernel(const Point &x, const float &maxDist2);
      inline virtual ~ConstantKernel(void){;};
      virtual float operator()(const Photon &p, const float d2) const;
    }; // end ConstantKernel

    virtual void evaluateIndirect(const Scene &scene,
                                  const Path &p,
                                  std::vector<Result> &results) const;

    virtual Spectrum estimateRadiance(const Vector &wo,
                                      const DifferentialGeometry &dg,
                                      const ScatteringDistributionFunction *f,
                                      const float maxDist2,
                                      const PhotonGatherer &gather) const;

    boost::shared_ptr<const PhotonMap> mGlobalMap;
    boost::shared_ptr<const PhotonMap> mCausticMap;

    /*! The number of final gather rays, in "x" and "y"
     */
    size_t mFinalGatherX, mFinalGatherY;

    /*! This caches 1 / (mFinalGatherX * mFinalGatherY)
     */
    float mInvNumStrata;

    /*! This thing produces stratified hemisphere samples.
     *  XXX This is made a mutable member to avoid the expense
     *      of making it a stack variable in evaluate().
     */
    mutable HilbertSequence mStratifiedSequence;

    /*! This thing gathers photons.
     *  XXX This is made a mutable member to avoid the expense
     *      of making it a stack variable in evaluate().
     */
    mutable PhotonGatherer mGather;
}; // end JensenSampler

#endif // JENSEN_SAMPLER_H

