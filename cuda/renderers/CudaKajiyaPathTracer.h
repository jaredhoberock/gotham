/*! \file CudaKajiyaPathTracer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a CudaRenderer
 *         which performs stateless Kajiya-style
 *         path tracing.
 */

#pragma once

#include "CudaRenderer.h"
#include <stdcuda/device_types.h>
#include <vector_types.h>

class CudaKajiyaPathTracer
  : public CudaRenderer
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef CudaRenderer Parent;

    /*! Null constructor calls the Parent.
     */
    inline CudaKajiyaPathTracer(void);

    /*! Constructor accepts a pointer to a Scene and Record.
     *  \param s Sets mScene.
     *  \param r Sets mRecord.
     */
    inline CudaKajiyaPathTracer(boost::shared_ptr<const Scene> s,
                                boost::shared_ptr<Record> r);

  protected:
    virtual void kernel(ProgressCallback &progress);

    virtual void generateHyperPoints(const stdcuda::device_ptr<float4> &u,
                                     const size_t n);
    virtual void stratify(const size_t w, const size_t h,
                          std::vector<float4> &u);
    virtual void stratify(const size_t w, const size_t h,
                          const stdcuda::device_ptr<float4> &u,
                          const size_t n);
}; // end CudaKajiyaPathTracer

#include "CudaKajiyaPathTracer.inl"

