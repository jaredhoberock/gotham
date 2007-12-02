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

    /*! Constructor takes a shared_pointer to a RenderFilm containing an estimate.
     *  \param estimate Sets mEstimate.
     */
    EstimateImportance(const boost::shared_ptr<RenderFilm> &estimate);

    /*! This method converts the spectral Monte Carlo throughput
     *  of a Path into scalar importance.
     *  \param x The HyperPoint uniquely specifying the Path of interest.
     *  \param f The spectral Monte Carlo throughput of the Path of interest.
     *  \return The scalar importance of x.
     */
    virtual float evaluate(const PathSampler::HyperPoint &x,
                           const Path &xPath,
                           const std::vector<PathSampler::Result> &results);

  protected:
    /*! A pointer to the estimate of the image to render.
     */
    boost::shared_ptr<RenderFilm> mEstimate;

    /*! This maps a Path to an image location.
     */
    PathToImage mMapToImage;

    // XXX TODO create an array of 1/luminance
}; // end EstimateImportance

#endif // ESTIMATE_IMPORTANCE_H

