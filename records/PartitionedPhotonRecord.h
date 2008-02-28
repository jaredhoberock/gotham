/*! \file PartitionedPhotonRecord.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a PhotonRecord which
 *         partitions Photons into "global" and "caustic"
 *         PhotonMaps.
 */

#ifndef PARTITIONED_PHOTON_RECORD_H
#define PARTITIONED_PHOTON_RECORD_H

#include "PhotonRecord.h"

class PartitionedPhotonRecord
  : public PhotonRecord
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef PhotonRecord Parent;

    /*! Null constructor does nothing.
     */
    PartitionedPhotonRecord(void);

    /*! Constructor calls the Parent.
     *  \param n The number of Photons to reserve.
     *  \param filename A filename to write to post-rendering.
     */
    PartitionedPhotonRecord(const size_t n,
                            const std::string &filename = "");

    /*! This method deposits Photons along xPath to this PhotonRecord.
     *  \param w The weight to associate with the record.
     *  \param x The HyperPoint associated with xPath.
     *  \param xPath The Path to record.
     *  \param results A list of PathSampler::Results to record.
     */
    virtual void record(const float w,
                        const PathSampler::HyperPoint &x,
                        const Path &xPath,
                        const std::vector<PathSampler::Result> &results);

    /*! This method is called after rendering.
     */
    virtual void postprocess(void);

  protected:
    // Keep a "caustic" photon map separately
    PhotonMap mCausticMap;
}; // end PartitionedPhotonRecord

#endif // PARTITIONED_PHOTON_RECORD_H

