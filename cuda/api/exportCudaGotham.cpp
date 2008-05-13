/*! \file exportCudaGotham.cpp
 *  \author Jared Hoberock
 *  \brief Exports CudaGotham to Python.
 */

#include "CudaGotham.h"
#include <boost/python.hpp>
using namespace boost::python;

void exportCudaGotham(void)
{
  class_<CudaGotham>("CudaGotham")
    .def(init<const Gotham&>())
    .def("render", &CudaGotham::render)
    ;
} // end exportCudaGotham()

BOOST_PYTHON_MODULE(libcudagotham)
{
  exportCudaGotham();
} // end BOOST_PYTHON_MODULE()

