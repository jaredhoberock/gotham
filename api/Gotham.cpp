/*! \file Gotham.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of the Gotham class.
 */

#include "Gotham.h"
#include <boost/shared_ptr.hpp>

#include "../shading/exportShading.h"
#include "../shading/Material.h"
#include "../shading/DefaultMaterial.h"
#include "../shading/stdshader.h"
#include "../surfaces/Mesh.h"
#include "../surfaces/SmallMesh.h"
#include "../surfaces/Sphere.h"
#include "../primitives/SurfacePrimitive.h"
#include "../primitives/PrimitiveBSP.h"
#include "../viewers/RenderViewer.h"
#include "../renderers/RendererApi.h"
#include "../records/RenderFilm.h"
#include "../records/GpuFilm.h"
#include "../rasterizables/RasterizableScene.h"
#include "../rasterizables/RasterizablePrimitiveList.h"
#include "../rasterizables/RasterizableSurfacePrimitive.h"
#include "../rasterizables/RasterizableMesh.h"
#include "../rasterizables/RasterizableSphere.h"
#include "../primitives/UnshadowedScene.h"
#pragma warning(push)
#pragma warning(disable : 4311 4312)
#include <Qt/qapplication.h>
#pragma warning(pop)

using namespace boost;

Gotham
  ::Gotham(void)
{
  init();
} // end Gotham::Gotham()

Gotham
  ::~Gotham(void)
{
  ;
} // end Gotham::~Gotham()

void Gotham
  ::init(void)
{
  // XXX HACK this is completely retarded, but
  //     noise() won't get exported to gotham.lib
  //     unless we actually use it somewhere in the code that gets compiled
  //     into the dll
  //     thanks msvc!
  float x = gotham::noise(0,0,0);

  // clear the Matrix stack
  mMatrixStack.clear();

  // push an identity Matrix
  mMatrixStack.push_back(Matrix::identity());

  // clear the attribute stack
  mAttributeStack.clear();

  // push default attributes
  AttributeMap attr;
  mAttributeStack.resize(1);
  getDefaultAttributes(mAttributeStack.back());

  // create the PrimitiveList
  mPrimitives.reset(new RasterizablePrimitiveList< PrimitiveBSP<> >());

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

  if(mMatrixStack.empty())
  {
    std::cerr << "Gotham::popMatrix(): Warning, matrix stack empty!" << std::endl;
    mMatrixStack.push_back(Matrix::identity());
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
  // create a new Scene
  AttributeMap::const_iterator a = mAttributeStack.back().find("scene::castshadows");
  shared_ptr<Scene> s;
  if(any_cast<std::string>(a->second) == std::string("false"))
  {
    s.reset(new RasterizableScene<UnshadowedScene>());
  } // end if
  else
  {
    s.reset(new RasterizableScene<Scene>());
  } // end else

  // finalize all primitives
  mPrimitives->finalize();
  mEmitters->finalize();
  mSensors->finalize();

  // hand over the primitives
  s->setPrimitive(mPrimitives);

  // give the lights to the scene
  s->setEmitters(mEmitters);

  // give the sensors to the scene
  s->setSensors(mSensors);

  // create a new Renderer
  mRenderer.reset(RendererApi::renderer(mAttributeStack.back()));

  // name of the output?
  std::string outfile = "";
  a = mAttributeStack.back().find("renderer::outfile");
  if(a != mAttributeStack.back().end())
  {
    any val = a->second;
    outfile = any_cast<std::string>(val).c_str();
  } // end if

  // should we normalize on post?
  bool doNormalize = false;
  a = mAttributeStack.back().find("renderer::normalize");
  if(a != mAttributeStack.back().end())
  {
    any val = a->second;
    doNormalize = (any_cast<std::string>(val) == std::string("true"));
  } // end if

  // give everything to the Renderer
  mRenderer->setScene(s);
  shared_ptr<RenderFilm> film(new RenderFilm(width,height,outfile));
  //shared_ptr<RenderFilm> film(new GpuFilm<RenderFilm>());
  //shared_ptr<RenderFilm> film(new GpuFilterFilm<RenderFilm>());
  film->resize(width,height);
  film->setFilename(outfile);
  film->setNormalizeOnPostprocess(doNormalize);
  mRenderer->setFilm(film);

  // headless render?
  bool headless = false;
  a = mAttributeStack.back().find("viewer");
  if(a != mAttributeStack.back().end())
  {
    any val = a->second;
    headless = (any_cast<std::string>(val) != std::string("true"));
  } // end if

  if(!headless)
  {
    int zero = 0;
    QApplication application(zero,0);

    RenderViewer v;
    v.resize(width, height);

    // everything to the viewer
    v.setScene(s);
    v.setImage(film);
    v.setRenderer(mRenderer);

    v.setSnapshotFileName(mRenderer->getRenderParameters().c_str());

    // try to tell the viewer where to look
    // bail out otherwise
    try
    {
      float fovy = atof(any_cast<std::string>(mAttributeStack.back()["viewer::fovy"]).c_str());
      float eyex = atof(any_cast<std::string>(mAttributeStack.back()["viewer::eyex"]).c_str());
      float eyey = atof(any_cast<std::string>(mAttributeStack.back()["viewer::eyey"]).c_str());
      float eyez = atof(any_cast<std::string>(mAttributeStack.back()["viewer::eyez"]).c_str());
      float upx  = atof(any_cast<std::string>(mAttributeStack.back()["viewer::upx"]).c_str());
      float upy  = atof(any_cast<std::string>(mAttributeStack.back()["viewer::upy"]).c_str());
      float upz  = atof(any_cast<std::string>(mAttributeStack.back()["viewer::upz"]).c_str());
      float lookx  = atof(any_cast<std::string>(mAttributeStack.back()["viewer::lookx"]).c_str());
      float looky  = atof(any_cast<std::string>(mAttributeStack.back()["viewer::looky"]).c_str());
      float lookz  = atof(any_cast<std::string>(mAttributeStack.back()["viewer::lookz"]).c_str());

      // convert degrees to radians
      float fovyRadians = fovy * PI / 180.0f;
      v.camera()->setFieldOfView(fovyRadians);
      v.camera()->setAspectRatio(float(width)/float(height));
      v.camera()->setPosition(qglviewer::Vec(eyex,eyey,eyez));
      v.camera()->setUpVector(qglviewer::Vec(upx,upy,upz));
      v.camera()->setViewDirection(qglviewer::Vec(lookx,looky,lookz));
    } // end try
    catch(...)
    {
      ;
    } // end catch

    v.show();

    application.exec();
  } // end if
  else
  {
    // start rendering
    Renderer::ProgressCallback callback;
    mRenderer->render(callback);
  } // end else

  return;
} // end Gotham::render()

void Gotham
  ::material(Material *m)
{
  mAttributeStack.back()["material"] = shared_ptr<Material>(m);
} // end Gotham::material)

void Gotham
  ::sphere(const float cx,
           const float cy,
           const float cz,
           const float radius)
{
  // XXX should we scale the radius?
  Point c(cx,cy,cz);
  c = mMatrixStack.back()(c);

  shared_ptr<Surface> surface(new RasterizableSphere(c, radius));
  shared_ptr<Material> m = any_cast<shared_ptr<Material> >(mAttributeStack.back()["material"]);
  surfacePrimitive(new RasterizableSurfacePrimitive(surface, m));
} // end Gotham::sphere()

void Gotham
  ::surfacePrimitive(SurfacePrimitive *prim)
{
  // name the primitive
  prim->setName(any_cast<std::string>(mAttributeStack.back()["name"]));

  shared_ptr<SurfacePrimitive> surfacePrim(prim);
  shared_ptr<Primitive> plainPrim = static_pointer_cast<Primitive,SurfacePrimitive>(surfacePrim);
  mPrimitives->push_back(plainPrim);

  if(prim->getMaterial()->isEmitter())
  {
    mEmitters->push_back(surfacePrim);
  } // end if

  if(prim->getMaterial()->isSensor())
  {
    mSensors->push_back(surfacePrim);
  } // end if
} // end Gotham::surfacePrimitive()

void Gotham
  ::mesh(std::vector<float> &vertices,
         std::vector<unsigned int> &faces)
{
  // do we need to reverse the winding of vertices?
  bool reverse = any_cast<std::string>(mAttributeStack.back()["orientation"]) == "inside";

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
    if(reverse)
    {
      triangles.push_back(Mesh::Triangle(faces[i], faces[i+2], faces[i+1]));
    } // end if
    else
    {
      triangles.push_back(Mesh::Triangle(faces[i], faces[i+1], faces[i+2]));
    } // end if
  } // end for i

  Mesh *mesh = 0;
  if(faces.size() > 15)
  {
    mesh = new RasterizableMesh<Mesh>(points, triangles);
  } // end if
  else
  {
    mesh = new RasterizableMesh<SmallMesh>(points, triangles);
  } // end else

  shared_ptr<Surface> surface(mesh);
  shared_ptr<Material> m = any_cast<shared_ptr<Material> >(mAttributeStack.back()["material"]);
  surfacePrimitive(new RasterizableSurfacePrimitive(surface, m)); 
} // end Gotham::mesh()

void Gotham
  ::mesh(std::vector<float> &vertices,
         std::vector<float> &parametrics,
         std::vector<unsigned int> &faces)
{
  // do we need to reverse the winding of vertices?
  bool reverse = any_cast<std::string>(mAttributeStack.back()["orientation"]) == "inside";

  std::vector<Point> points;
  std::vector<ParametricCoordinates> parms;
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
      i != parametrics.size();
      i += 2)
  {
    parms.push_back(ParametricCoordinates(parametrics[i], parametrics[i+1]));
  } // end for i

  for(unsigned int i = 0;
      i != faces.size();
      i += 3)
  {
    if(reverse)
    {
      triangles.push_back(Mesh::Triangle(faces[i], faces[i+2], faces[i+1]));
    } // end if
    else
    {
      triangles.push_back(Mesh::Triangle(faces[i], faces[i+1], faces[i+2]));
    } // end if
  } // end for i

  Mesh *mesh = 0;
  if(faces.size() > 15)
  {
    mesh = new RasterizableMesh<Mesh>(points, parms, triangles);
  } // end if
  else
  {
    mesh = new RasterizableMesh<SmallMesh>(points, parms, triangles);
  } // end else

  shared_ptr<Surface> surface(mesh);
  shared_ptr<Material> m = any_cast<shared_ptr<Material> >(mAttributeStack.back()["material"]);
  surfacePrimitive(new RasterizableSurfacePrimitive(surface, m)); 
} // end Gotham::mesh()

void Gotham
  ::getDefaultAttributes(AttributeMap &attr) const
{
  attr.clear();

  any toAdd = std::string("kajiya");
  attr["path::sampler"] = toAdd;

  toAdd = std::string("4");
  attr["path::maxlength"] = toAdd;

  toAdd = std::string("4");
  attr["renderer::spp"] = toAdd;

  toAdd = std::string("true");
  attr["viewer"] = toAdd;

  toAdd = std::string("gotham.exr");
  attr["renderer::outfile"] = toAdd;

  toAdd = std::string("");
  attr["name"] = toAdd;

  toAdd = shared_ptr<Material>(new DefaultMaterial());
  attr["material"] = toAdd;

  toAdd = std::string("true");
  attr["scene::castshadows"] = toAdd;

  // by default, assume normals point towards the outside
  // of the object
  toAdd = std::string("outside");
  attr["orientation"] = toAdd;
} // end Gotham::getDefaultAttributes()

void Gotham
  ::attribute(const std::string &name,
              const std::string &val)
{
  any toAdd = val;
  mAttributeStack.back()[name] = toAdd;
} // end Gotham::attribute()

void Gotham
  ::pushAttributes(void)
{
  mAttributeStack.push_back(mAttributeStack.back());
} // end Gotham::pushAttributes()

void Gotham
  ::popAttributes(void)
{
  mAttributeStack.pop_back();
} // end Gotham::popAttributes()

