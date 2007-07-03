/*! \file Gotham.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of the Gotham class.
 */

#include "Gotham.h"

#include <GL/glew.h>
#include <QGLViewer/qglviewer.h>
#include <viewermain/main.h>
#include <commonviewer/CommonViewer.h>
#include <QtGui/QKeyEvent>

Gotham
  ::Gotham(void)
{
  init();
} // end Gotham::Gotham()

void Gotham
  ::init(void)
{
  // clear the Matrix stack
  mMatrixStack.clear();

  // push an identity Matrix
  mMatrixStack.push_back(Matrix::identity());
} // end Gotham:init()

void Gotham
  ::pushMatrix(void)
{
  mMatrixStack.push_back(mMatrixStack.back());
} // end Gotham::pushMatrix()

void Gotham
  ::popMatrix(void)
{
  if(mMatrixStack.size() > 1)
  {
    mMatrixStack.pop_back();
  } // end if
  else
  {
    std::cerr << "Gotham::popMatrix(): Warning, matrix stack empty!" << std::endl;
  } // end else
} // end Gotham::popMatrix()

void Gotham
  ::loadMatrix(const Matrix &m)
{
  mMatrixStack.back() = m;
} // end Gotham::loadMatrix()

void Gotham
  ::render(void)
{
  std::cerr << "Gotham::render(): Entered." << std::endl;

  // start a viewer
  viewerMain<CommonViewer<QGLViewer, QKeyEvent> >(0, 0);
} // end Gotham::render()

void Gotham
  ::material(const char *name)
{
//#ifdef WIN32
//  std::cerr << "Gotham::material(): Implement me!" << std::endl;
//#else
//  hinstLib = dlopen(fname.c_str(), RTLD_LAZY);
//  if(!hinstLib)
//  {
//    std::cerr << "Gotham::material(): Can't
//  } // end if
//#endif // WIN32
} // end Gotham::material()

#include <boost/python.hpp>
using namespace boost::python;
BOOST_PYTHON_MODULE(gotham)
{
  class_<Gotham>("Gotham")
    .def("pushMatrix", &Gotham::pushMatrix)
    .def("popMatrix", &Gotham::popMatrix)
    .def("render", &Gotham::render)
  ;
} // end gotham

