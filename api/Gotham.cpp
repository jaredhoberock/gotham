/*! \file Gotham.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of the Gotham class.
 */

#include "Gotham.h"
#include <boost/shared_ptr.hpp>

#include "../shading/Material.h"
#include "../surfaces/Mesh.h"
#include "../primitives/SurfacePrimitive.h"
#include "../viewers/RenderViewer.h"
#include "../renderers/DebugRenderer.h"
#include "../renderers/LightDebugRenderer.h"
#include "../films/RandomAccessFilm.h"
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

  // by default, we create a DebugRenderer
  //mRenderer.reset(new DebugRenderer());
  mRenderer.reset(new LightDebugRenderer());

  // create the PrimitiveList
  mPrimitives.reset(new PrimitiveList<>());

  // create the emitters list
  mEmitters.reset(new SurfacePrimitiveList());

  // create the sensors list
  mSensors.reset(new SurfacePrimitiveList());
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
  ::translate(const float tx, const float ty, const float tz)
{
  multMatrix(Matrix::translate(tx,ty,tz));
} // end Gotham::translate()

void Gotham
  ::rotate(const float degrees, const float rx, const float ry, const float rz)
{
  multMatrix(Matrix::rotate(degrees, rx, ry, rz));
} // end Gotham::rotate()

void Gotham
  ::scale(const float sx, const float sy, const float sz)
{
  multMatrix(Matrix::scale(sx,sy,sz));
} // end Gotham::scale()

void Gotham
  ::multMatrix(const std::vector<float> &m)
{
  multMatrix(Matrix(m[ 0], m[ 1], m[ 2], m[ 3],
                    m[ 4], m[ 5], m[ 6], m[ 7],
                    m[ 8], m[ 9], m[10], m[11],
                    m[12], m[13], m[14], m[15]));
} // end Gotham::multMatrix()

void Gotham
  ::loadMatrix(const std::vector<float> &m)
{
  loadMatrix(Matrix(m[ 0], m[ 1], m[ 2], m[ 3],
                    m[ 4], m[ 5], m[ 6], m[ 7],
                    m[ 8], m[ 9], m[10], m[11],
                    m[12], m[13], m[14], m[15]));
} // end Gotham::loadMatrix()

void Gotham
  ::getMatrix(std::vector<float> &m)
{
  const Matrix &top = mMatrixStack.back();
  m.resize(16);
  m[0]  = top.first(0,0); m[1]  = top.first(0,1); m[2]  = top.first(0,2), m[3]  = top.first(0,3);
  m[4]  = top.first(1,0); m[5]  = top.first(1,1); m[6]  = top.first(1,2), m[7]  = top.first(1,3);
  m[8]  = top.first(2,0); m[9]  = top.first(2,1); m[10] = top.first(2,2); m[11] = top.first(2,3);
  m[12] = top.first(3,0); m[13] = top.first(3,1); m[14] = top.first(3,2); m[15] = top.first(3,3);
} // end Gotham::getMatrix()

void Gotham
  ::loadMatrix(const Matrix &m)
{
  mMatrixStack.back() = m;
} // end Gotham::loadMatrix()

void Gotham
  ::multMatrix(const Matrix &m)
{
  Matrix &top = mMatrixStack.back();
  top = top * m;
} // end Gotham::multMatrix()

void Gotham
  ::render(const unsigned int width,
           const unsigned int height)
{
  QApplication application(0,0);

  // create a new Scene
  boost::shared_ptr<Scene> s(new Scene());

  RenderViewer v;
  v.resize(width, height);

  // give it the primitives
  s->setPrimitive(mPrimitives);

  // finalize emitters & sensors
  mEmitters->finalize();
  mSensors->finalize();

  // give the lights to the scene
  s->setEmitters(mEmitters);

  // give the sensors to the scene
  s->setSensors(mSensors);

  // give everything to the Renderer
  mRenderer->setScene(s);
  shared_ptr<RandomAccessFilm> film(new RandomAccessFilm(width,height));
  mRenderer->setFilm(film);

  // everything to the viewer
  v.setScene(s);
  v.setImage(film);
  v.setRenderer(mRenderer);

  v.show();

  application.exec();

  return;
} // end Gotham::render()

void Gotham
  ::material(Material *m)
{
  mCurrentMaterial = boost::shared_ptr<Material>(m);
} // end Gotham::material)

void Gotham
    ::mesh(std::vector<float> &vertices,
           std::vector<unsigned int> &faces)
{
  std::vector<Point> points;
  std::vector<Mesh::Triangle> triangles;
  for(unsigned int i = 0;
      i != vertices.size();
      i += 3)
  {
    points.push_back(Point(vertices[i], vertices[i+1], vertices[i+2]));

    // transform by the matrix on the top of the stack
    points.back() = mMatrixStack.back()(points.back());
  } // end for i

  for(unsigned int i = 0;
      i != faces.size();
      i += 3)
  {
    triangles.push_back(Mesh::Triangle(faces[i], faces[i+1], faces[i+2]));
  } // end for i

  // XXX create a common primitive() or surfacePrimitive() call to pass
  //     to here
  Mesh *mesh = new Mesh(points, triangles);
  boost::shared_ptr<Surface> surface(mesh);
  boost::shared_ptr<SurfacePrimitive> surfacePrim(new SurfacePrimitive(surface, mCurrentMaterial));
  boost::shared_ptr<Primitive> prim = static_pointer_cast<Primitive,SurfacePrimitive>(surfacePrim);
  mPrimitives->push_back(prim);

  if(mCurrentMaterial->isEmitter())
  {
    mEmitters->push_back(surfacePrim);
  } // end if

  if(mCurrentMaterial->isSensor())
  {
    mSensors->push_back(surfacePrim);
  } // end if
} // end Gotham::mesh()

// wrapper for Gotham::material()
void Gotham_material(Gotham &g, std::auto_ptr<Material> m)
{
  g.material(m.get());
  m.release();
} // end Gotham_material()

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
using namespace boost::python;
BOOST_PYTHON_MODULE(gotham)
{
  // tell boost which multMatrix we mean
  typedef void (Gotham::*multMatrix_vector)(const std::vector<float>&);

  // tell boost which loadMatrix we mean
  typedef void (Gotham::*loadMatrix_vector)(const std::vector<float>&);

  // tell boost which getMatrix we mean
  typedef void (Gotham::*getMatrix_vector)(const std::vector<float>&);

  class_<Gotham>("Gotham")
    .def("pushMatrix", &Gotham::pushMatrix)
    .def("popMatrix", &Gotham::popMatrix)
    .def("translate", &Gotham::translate)
    .def("rotate", &Gotham::rotate)
    .def("scale", &Gotham::scale)
    .def("multMatrix", multMatrix_vector(&Gotham::multMatrix))
    .def("loadMatrix", loadMatrix_vector(&Gotham::loadMatrix))
    .def("getMatrix", getMatrix_vector(&Gotham::getMatrix))
    .def("mesh", &Gotham::mesh)
    .def("render", &Gotham::render)
    .def("material", Gotham_material)
    ;

  class_<Material, std::auto_ptr<Material> >("Material")
    ;

  class_<std::vector<float> >("vector_float")
    .def(vector_indexing_suite<std::vector<float> >())
    ;

  class_<std::vector<uint> >("vector_uint")
    .def(vector_indexing_suite<std::vector<unsigned int> >())
    ;
} // end gotham

