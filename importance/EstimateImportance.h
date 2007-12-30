/*! \file EstimateImportance.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an ScalarImportance class
 *         which assigns importance based on an estimate of the image to render.
 */

#ifndef ESTIMATE_IMPORTANCE_H
#define ESTIMATE_IMPORTANCE_H

#include "ScalarImportance.h"
#include "../path/PathToImage.h"
#include "../records/RenderFilm.h"

class EstimateImportance
  : public ScalarImportance
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScalarImportance Parent;

    /*! Constructor takes a reference to a RandomAccessFilm containing an estimate.
     *  \param estimate Sets mEstimate.
     */
    EstimateImportance(const RandomAccessFilm &estimate);

    /*! This method converts the spectral Monte Carlo throughput
     *  of a Path into scalar importance.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param f The spectral Monte Carlo throughput of the Path of interest.
     *  \return The scalar importance of x.
     */
    virtual float evaluate(const PathSampler::HyperPoint &x,
                           const Path &xPath,
                           const std::vector<PathSampler::Result> &results);

    /*! This method converts the spectral Monte Carlo throughput
     *  of a Path into scalar importance.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param xPath The Path of interest.
     *  \param r The Result of interest.
     *  \return The scalar importance of r.
     */
    virtual float evaluate(const PathSampler::HyperPoint &x,
                           const Path &xPath,
                           const PathSampler::Result &r) const;

  protected:
    /*! This maps a Path to an image location.
     */
    PathToImage mMapToImage;

    RandomAccessFilm mEstimate;
}; // end EstimateImportance

#endif // ESTIMATE_IMPORTANCE_H

