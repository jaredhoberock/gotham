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

#include "../shading/Material.h"

#ifndef WIN32
#include <dlfcn.h>
#endif // WIN32

Gotham
  ::Gotham(void)
{
  init();
} // end Gotham::Gotham()

Gotham
  ::~Gotham(void)
{
  std::cerr << "Gotham::~Gotham()." << std::endl;
} // end Gotham::~Gotham()

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
  if(!mMatrixStack.empty())
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

bool Gotham
  ::material(const char *name)
{
  // deal with paths here?
  Material *m = loadMaterial(name);
  if(m)
  {
    mCurrentMaterial.reset(m);
    return true;
  } // end if

  return false;
} // end Gotham::material()

Material *Gotham
  ::loadMaterial(const char *path)
{
  Material *result = 0;

#ifdef WIN32
  std::cerr << "Gotham::loadMaterial(): Implement me!" << std::endl;
#else
  void *handle = dlopen(path, RTLD_LAZY);
  if(!handle)
  {
    std::cerr << "Gotham::loadMaterial(): Couldn't find Material " << path << std::endl;
  } // end if
  else
  {
    // reset errors
    dlerror();
    typedef Material *(*CreateMaterial)();
    CreateMaterial createMaterial = reinterpret_cast<CreateMaterial>(dlsym(handle, "createMaterial"));
    const char *error = dlerror();
    if(!error)
    {
      // try calling it
      result = createMaterial();
    } // end if

    // close the dll
    dlclose(handle);
  } // end else
#endif // WIN32

  return result;
} // end Gotham::loadMaterial()

#include <boost/python.hpp>
using namespace boost::python;
BOOST_PYTHON_MODULE(gotham)
{
  class_<Gotham>("Gotham")
    .def("pushMatrix", &Gotham::pushMatrix)
    .def("popMatrix", &Gotham::popMatrix)
    .def("material", &Gotham::material)
    .def("render", &Gotham::render)
  ;
} // end gotham

