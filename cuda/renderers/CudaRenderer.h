/*! \file CudaRenderer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a SIMD renderer which
 *         runs on a CUDA device.
 */

#pragma once

#include "../../renderers/SIMDRenderer.h"

class CudaRenderer
  : public SIMDRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef SIMDRenderer Parent;

    /*! Null constructor calls the Parent.
     */
    inline CudaRenderer(void);

    /*! Constructor accpepts a pointer to a Scene and Record.
     *  \param s Sets mScene.
     *  \param r Sets mRecord.
     */
    inline CudaRenderer(boost::shared_ptr<const Scene> s,
                        boost::shared_ptr<Record> r);

    /*! This method sets mScene.
     *  \param s Sets mScene.
     *  \note s must be a CudaScene.
     */
    inline virtual void setScene(const boost::shared_ptr<const Scene> &s);

    /*! This method sets mShadingContext.
     *  \param s Sets mShadingContext.
     *  \note s must be a CudaShadingContext.
     */
    inline virtual void setShadingContext(const boost::shared_ptr<ShadingContext> &s);
}; // end CudaRenderer

#include "CudaRenderer.inl"

