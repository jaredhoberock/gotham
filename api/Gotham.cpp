/*! \file Gotham.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of the Gotham class.
 */

#include "Gotham.h"

#include "../shading/Material.h"
#include "../shading/MaterialViewer.h"
#pragma warning(push)
#pragma warning(disable : 4311 4312)
#include <Qt/qapplication.h>
#pragma warning(pop)

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

  QApplication application(0,0);

  // start a viewer
  MaterialViewer v;
  v.setMaterial(mCurrentMaterial);
  v.show();

  application.exec();
} // end Gotham::render()

bool Gotham
  ::material(const char *name)
{
  std::cerr << "Gotham::loadMaterial(): about to load." << std::endl;
  Material *m = loadMaterial(name);
  std::cerr << "Gotham::loadMaterial(): after load." << std::endl;
  if(m)
  {
    std::cerr << "Gotham::material(): able to load Material " << m->getName() << std::endl;
    mCurrentMaterial = m;
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

    // XXX figure out when its safe to close the dll
    //// close the dll
    //dlclose(handle);
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

