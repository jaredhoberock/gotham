/*! \file Gotham.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of the Gotham class.
 */

#include "Gotham.h"
#include <boost/shared_ptr.hpp>
#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>

#include "../shading/exportShading.h"
#include "../shading/Material.h"
#include "../shading/DefaultMaterial.h"
#include "../shading/stdshader.h"
#include "../surfaces/Mesh.h"
#include "../surfaces/SmallMesh.h"
#include "../surfaces/Sphere.h"
#include "../primitives/SurfacePrimitive.h"
#include "../primitives/PrimitiveBSP.h"
#include "../primitives/TriangleBVH.h"
#include "../viewers/RenderViewer.h"
#include "../renderers/RendererApi.h"
#include "../records/RecordApi.h"
#include "../rasterizables/RasterizableScene.h"
#include "../rasterizables/RasterizablePrimitiveList.h"
#include "../rasterizables/RasterizableSurfacePrimitive.h"
#include "../rasterizables/RasterizableMesh.h"
#include "../rasterizables/RasterizableSphere.h"
#include "../primitives/UnshadowedScene.h"

// APIs
#include "../path/PathApi.h"
#include "../mutators/MutatorApi.h"
#include "../importance/ImportanceApi.h"
#include "../renderers/RendererApi.h"

#pragma warning(push)
#pragma warning(disable : 4311 4312)
#include <Qt/qapplication.h>
#pragma warning(pop)

using namespace boost;
using namespace boost::spirit;

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
  x += x;

  // clear the Matrix stack
  mMatrixStack.clear();

  // push an identity Matrix
  mMatrixStack.push_back(Matrix::identity());

  // clear the attribute stack
  mAttributeStack.clear();

  // clear materials
  mMaterials.clear();

  // create the default material
  mMaterials.push_back(shared_ptr<Material>(new DefaultMaterial()));

  // push default attributes
  AttributeMap attr;
  mAttributeStack.resize(1);
  getDefaultAttributes(mAttributeStack.back());

  // create the PrimitiveList
  //mPrimitives.reset(new RasterizablePrimitiveList< PrimitiveBSP<> >());
  mPrimitives.reset(new RasterizablePrimitiveList< TriangleBVH >());

  // create the emitters list
  mEmitters.reset(new RasterizablePrimitiveList< SurfacePrimitiveList >());

  // create the sensors list
  mSensors.reset(new RasterizablePrimitiveList< SurfacePrimitiveList >());

  // clear the PhotonMap list
  mPhotonMaps.clear();
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
  ::render(void)
{
  AttributeMap &attr = mAttributeStack.back();

  // create a new Scene
  shared_ptr<Scene> s;
  if(attr["scene:castshadows"] == std::string("false"))
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
  mRenderer.reset(RendererApi::renderer(mAttributeStack.back(), mPhotonMaps));

  // create a Record
  shared_ptr<Record> record;
  record.reset(RecordApi::record(mAttributeStack.back()));

  // give everything to the renderer
  mRenderer->setScene(s);
  mRenderer->setRecord(record);

  // headless render?
  bool headless = (attr["viewer"] == std::string("false"));

  if(!headless)
  {
    int zero = 0;
    QApplication application(zero,0);

    RenderViewer v;

    // title the window the name of the outfile
    v.setWindowTitle(attr["record:outfile"].c_str());

    // everything to the viewer
    v.setScene(s);

    v.setRenderer(mRenderer);

    v.setSnapshotFileName(mRenderer->getRenderParameters().c_str());

    v.setGamma(lexical_cast<float>(attr["viewer:gamma"]));

    // try to tell the viewer where to look
    // bail out otherwise
    try
    {
      float fovy   = lexical_cast<float>(attr["viewer:fovy"]);
      float eyex   = lexical_cast<float>(attr["viewer:eyex"]);
      float eyey   = lexical_cast<float>(attr["viewer:eyey"]);
      float eyez   = lexical_cast<float>(attr["viewer:eyez"]);
      float upx    = lexical_cast<float>(attr["viewer:upx"]);
      float upy    = lexical_cast<float>(attr["viewer:upy"]);
      float upz    = lexical_cast<float>(attr["viewer:upz"]);
      float lookx  = lexical_cast<float>(attr["viewer:lookx"]);
      float looky  = lexical_cast<float>(attr["viewer:looky"]);
      float lookz  = lexical_cast<float>(attr["viewer:lookz"]);
      float width  = lexical_cast<float>(attr["record:width"]);
      float height = lexical_cast<float>(attr["record:height"]);

      // convert degrees to radians
      float fovyRadians = fovy * PI / 180.0f;
      v.camera()->setFieldOfView(fovyRadians);
      v.camera()->setAspectRatio(width/height);
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
  // add m to mMaterials
  mMaterials.push_back(shared_ptr<Material>(m));

  // note the current material
  mAttributeStack.back()["material"] = lexical_cast<std::string>(mMaterials.size() - 1);
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

  AttributeMap &attr = mAttributeStack.back();
  shared_ptr<Material> m = mMaterials[lexical_cast<size_t>(attr["material"])];
  surfacePrimitive(new RasterizableSurfacePrimitive(surface, m));
} // end Gotham::sphere()

void Gotham
  ::surfacePrimitive(SurfacePrimitive *prim)
{
  // name the primitive
  prim->setName(mAttributeStack.back()["name"]);

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
  bool reverse = mAttributeStack.back()["orientation"] == "inside";

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

  AttributeMap &attr = mAttributeStack.back();
  shared_ptr<Material> m = mMaterials[lexical_cast<size_t>(attr["material"])];
  surfacePrimitive(new RasterizableSurfacePrimitive(surface, m));
} // end Gotham::mesh()

void Gotham
  ::mesh(std::vector<float> &vertices,
         std::vector<float> &parametrics,
         std::vector<unsigned int> &faces)
{
  // do we need to reverse the winding of vertices?
  bool reverse = mAttributeStack.back()["orientation"] == "inside";

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

  AttributeMap &attr = mAttributeStack.back();
  shared_ptr<Material> m = mMaterials[lexical_cast<size_t>(attr["material"])];
  surfacePrimitive(new RasterizableSurfacePrimitive(surface, m));
} // end Gotham::mesh()

void Gotham
  ::getDefaultAttributes(AttributeMap &attr) const
{
  attr.clear();

  // call the other libraries to get their defaults
  ImportanceApi::getDefaultAttributes(attr);
  MutatorApi::getDefaultAttributes(attr);
  PathApi::getDefaultAttributes(attr);
  RecordApi::getDefaultAttributes(attr);
  RendererApi::getDefaultAttributes(attr);

  // set miscellaneous attributes that don't belong
  // elsewhere
  attr["viewer"] = "true";
  attr["viewer:gamma"] = "2.2";
  attr["name"] = "";
  attr["material"] = "0";
  attr["scene:castshadows"] = "true";
  attr["orientation"] = "outside";
} // end Gotham::getDefaultAttributes()

void Gotham
  ::attribute(const std::string &name,
              const std::string &val)
{
  mAttributeStack.back()[name] = val;
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

void Gotham
  ::photons(const std::vector<float> &positions,
            const std::vector<float> &wi,
            const std::vector<float> &power)
{
  std::vector<Point> points;
  std::vector<Vector> vectors;
  std::vector<Spectrum> spectrums;
  for(unsigned int i = 0;
      i != positions.size();
      i += 3)
  {
    points.push_back(Point(positions[i], positions[i+1], positions[i+2]));
  } // end for i

  for(unsigned int i = 0;
      i != wi.size();
      i += 3)
  {
    vectors.push_back(Vector(wi[i], wi[i+1], wi[i+2]));
  } // end for i

  for(unsigned int i = 0;
      i != power.size();
      i += 3)
  {
    spectrums.push_back(Spectrum(power[i], power[i+1], power[i+2]));
  } // end for i

  shared_ptr<PhotonMap> pm(new PhotonMap(points, vectors, spectrums));

  // sort the photon map
  pm->sort();

  std::string name = mAttributeStack.back()["name"];
  mPhotonMaps.insert(std::make_pair(name, pm));
} // end Gotham::photons()

bool Gotham
  ::parsePhotons(const std::string &line)
{
  std::vector<float> points;
  std::vector<float> vectors;
  std::vector<float> powers;

  rule<phrase_scanner_t> pointsRule  = (ch_p('(') | ch_p('[')) >> real_p[push_back_a(points)]  >> *(',' >> real_p[push_back_a(points)])  >> (ch_p(')') | ch_p(']'));
  rule<phrase_scanner_t> vectorsRule = (ch_p('(') | ch_p('[')) >> real_p[push_back_a(vectors)] >> *(',' >> real_p[push_back_a(vectors)]) >> (ch_p(')') | ch_p(']'));
  rule<phrase_scanner_t> powersRule  = (ch_p('(') | ch_p('[')) >> real_p[push_back_a(powers)]  >> *(',' >> real_p[push_back_a(powers)])  >> (ch_p(')') | ch_p(']'));

  bool result = parse(line.c_str(),
    // begin grammar
    (
      str_p("g.photons") >> '(' >> pointsRule >> ',' >> vectorsRule >> ',' >> powersRule >> ')' >> end_p
    )
    ,
    // end grammar
    
    space_p).full;

  if(result)
  {
    // instantiate the photon map
    photons(points, vectors, powers);
  } // end if

  return result;
} // end Gotham::parsePhotons()

bool Gotham
  ::parseLine(const std::string &line)
{
  bool result = false;
  if(parsePhotons(line))
  {
    result = true;
  } // end if

  return result;
} // end Gotham::parseLine()

