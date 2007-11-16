/*! \file Record.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an abstract class
 *         abstracting the idea of a record of a rendering
 *         computation.
 */

#ifndef RECORD_H
#define RECORD_H

#include "../path/Path.h"
#include "../path/PathSampler.h"

class Record
{
  public:
    /*! This method records the given weighted Path.
     *  \param w The weight to associate with the record.
     *  \param x The HyperPoint associated with xPath.
     *  \param xPath The Path to record.
     *  \param results A list of PathSampler::Results to record.
     *  \note This method must be implemented in a derived class.
     */
    virtual void record(const float w,
                        const PathSampler::HyperPoint &x,
                        const Path &xPath,
                        const std::vector<PathSampler::Result> &results) = 0;

    /*! This method is called prior to rendering.
     *  \note The default implementation of this method does nothing.
     */
    inline virtual void preprocess(void);

    /*! This method is called after rendering.
     *  \note The default implementation of this method does nothing.
     */
    inline virtual void postprocess(void);
}; // end Record

#include "Record.inl"

#endif // RECORD_H

