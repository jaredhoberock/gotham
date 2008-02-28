/*! \file PhotonRecord.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a PhotonMap
 *         which can be built while rendering.
 */

#ifndef PHOTON_RECORD_H
#define PHOTON_RECORD_H

#include "Record.h"
#include "PhotonMap.h"

class PhotonRecord
  : public Record,
    public PhotonMap
{
  public:
    /*! \typedef Parent0
     *  \brief Shorthand.
     */
    typedef Record Parent0;

    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef PhotonMap Parent1;

    /*! \typedef PhotonList
     *  \brief Shorthand.
     */
    typedef boost::array<Photon, Path::static_size> PhotonList;

    /*! Null constructor calls the parent.
     */
    PhotonRecord(void);

    /*! Constructor calls the parent and resets the statistics.
     *  \param n The number of Photons to reserve.
     *  \param filename A filename to write to post-rendering.
     */
    PhotonRecord(const size_t n,
                 const std::string &filename = "");

    /*! This method scales the Photons' power by the given factor.
     *  \param s The scale.
     */
    virtual void scale(const float s);

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

    /*! This method deposits Photons along a single PathSampler::Result to
     *  the Photons in the given list.
     *  \param w The weight to associate with the result.
     *  \param xPath The Path of interest.
     *  \param r The PathSampler::Result of interest.
     *  \param photons Photons associated with (xPath,r) will be accumulated
     *                 here. 
     */
    virtual void record(const float w,
                        const Path &xPath,
                        const PathSampler::Result &r,
                        PhotonList &photons) const;

    /*! This method deposits a new Photon with the given data.
     *  \param x The position of the Photon.
     *  \param wi The incoming direction of the Photon.
     *  \param p The power of the Photon.
     */
    void deposit(const Point &x,
                 const Vector &wi,
                 const Spectrum &p);

    /*! This method returns the number of deposits into this RenderFilm since
     *  the last resize() event.
     *  \return mNumDeposits.
     */
    size_t getNumDeposits(void) const;

    /*! This method is called after rendering.
     */
    virtual void postprocess(void);
    
    /*! This method returns the filename.
     *  \return mFilename
     */
    const std::string &getFilename(void) const;

    /*! This method sets mFilename.
     *  \param filename Sets mFilename.
     */
    void setFilename(const std::string &filename);

  protected:
    /*! This method resets the statistics.
     */
    void reset(void);

    /*! The number of times deposit() has been called
     *  since the last resize() event.
     */
    size_t mNumDeposits;

    /*! A filename to write to during postprocessing.
     */
    std::string mFilename;
}; // end PhotonRecord

#endif // PHOTON_RECORD_H

