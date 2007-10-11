/*! \file PathToImage.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a small class
 *         which maps a PathSampler::Result to a location
 *         in a camera's image plane.
 */

#ifndef PATH_TO_IMAGE_H
#define PATH_TO_IMAGE_H

#include "PathSampler.h"

class PathToImage
{
  public:
    /*! This method maps a PathSampler::Result to a location
     *  within the unit square understood to correspond to a point
     *  on a camera's image plane.
     *  \param r The Result of interest.
     *  \param x The HyperPoint which generated xPath.
     *  \param xPath The Path which resulted in r.
     *  \param u The u coordinate of xPath's projection onto the image plane
     *           is returned here.
     *  \param v The v coordinate of xPath's projection onto the image plane
     *           is returned here.
     */
    virtual void evaluate(const PathSampler::Result &r,
                          const PathSampler::HyperPoint &x,
                          const Path &xPath,
                          float &u,
                          float &v) const;

    /*! operator()() method calls evaluate.
     *  \param r The Result of interest.
     *  \param x The HyperPoint which generated xPath.
     *  \param xPath The Path which resulted in r.
     *  \param u The u coordinate of xPath's projection onto the image plane
     *           is returned here.
     *  \param v The v coordinate of xPath's projection onto the image plane
     *           is returned here.
     */
    void operator()(const PathSampler::Result &r,
                    const PathSampler::HyperPoint &x,
                    const Path &xPath,
                    float &u,
                    float &v) const;
}; // end PathToImage

#endif // PATH_TO_IMAGE_H

