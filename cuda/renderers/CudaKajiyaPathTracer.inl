/*! \file CudaKajiyaPathTracer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CudaKajiyaPathTracer.h.
 */

#include "CudaKajiyaPathTracer.h"

CudaKajiyaPathTracer
  ::CudaKajiyaPathTracer(void)
    :Parent()
{
  ;
} // end CudaKajiyaPathTracer::CudaKajiyaPathTracer()

CudaKajiyaPathTracer
  ::CudaKajiyaPathTracer(boost::shared_ptr<const Scene> s,
                         boost::shared_ptr<Record> r)
    :Parent(s,r)
{
  ;
} // end CudaKajiyaPathTracer::CudaKajiyaPathTracer()

