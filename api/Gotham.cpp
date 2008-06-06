/*! \file Gotham.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of the Gotham class.
 */

#include "Gotham.h"
#include <boost/shared_ptr.hpp>
#include <boost/spirit/core.hpp>
#include <boost/spirit/actor/push_back_actor.hpp>

#include "../include/exportShading.h"
#include "../shading/DefaultMaterial.h"
#include "../surfaces/Mesh.h"
#include "../surfaces/SmallMesh.h"
#include "../surfaces/Sphere.h"
#include "../primitives/SurfacePrimitive.h"
#include "../primitives/PrimitiveApi.h"
#include "../renderers/RendererApi.h"
#include "../records/RecordApi.h"
#include "../shading/ShadingApi.h"
#include "../shading/DeferredLoadTexture.h"
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

//#define USE_QGLVIEWER 0
#define USE_QGLVIEWER 1

// choose which viewer base class to use before this
// #include
#include "../viewers/RenderViewer.h"

#if USE_QGLVIEWER
#pragma warning(push)
#pragma warning(disable : 4311 4312)
#include <Qt/qapplication.h>
#pragma warning(pop)
#endif // USE_QGLVIEWER

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
  // clear the Matrix stack
  mMatrixStack.clear();

  // push an identity Matrix
  mMatrixStack.push_back(Matrix::identity());

  // clear the attribute stack
  mAttributeStack.clear();

  // clear materials
  mMaterials.reset(new MaterialList());

  // create the default material
  mMaterials->push_back(shared_ptr<Material>(new DefaultMaterial()));

  // clear textures
  mTextures.reset(new TextureList());

  // create the default texture
  mTextures->push_back(shared_ptr<Texture>(new Texture()));

  // push default attributes
  AttributeMap attr;
  mAttributeStack.resize(1);
  getDefaultAttributes(mAttributeStack.back());

  // create the PrimitiveList
  mPrimitives.reset(new RasterizablePrimitiveList<PrimitiveList>());

  // create the emitters list
  mEmitters.reset(new RasterizablePrimitiveList<SurfacePrimitiveList>());

  // create the sensors list
  mSensors.reset(new RasterizablePrimitiveList<SurfacePrimitiveList>());

  // create the surfaces list
  mSurfaces.reset(new SurfacePrimitiveList());

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
  shared_ptr<Scene> s(PrimitiveApi::scene(mAttributeStack.back()));

  // create a final PrimitiveList
  PrimitiveList *list = PrimitiveApi::list(mAttributeStack.back(),
                                           *mPrimitives);

  // set every primitive's PrimitiveHandle
  // XXX this really sucks but i can't find a better solution
  PrimitiveHandle h = 0;
  for(PrimitiveList::iterator prim = list->begin();
      prim != list->end();
      ++prim, ++h)
  {
    (*prim)->setPrimitiveHandle(h);
  } // end for i

  // hand over the primitives
  shared_ptr<PrimitiveList> listPtr(list);
  s->setPrimitive(listPtr);
  s->setPrimitives(listPtr);

  // give the surfaces to the scene
  s->setSurfaces(mSurfaces);

  // create a final SurfacePrimitiveList for emitters
  shared_ptr<SurfacePrimitiveList> emitters(PrimitiveApi::surfacesList(mAttributeStack.back(),
                                                                       *mEmitters));
  // give the emitters to the scene
  s->setEmitters(emitters);

  // create a final SurfacePrimitiveList for sensors
  shared_ptr<SurfacePrimitiveList> sensors(PrimitiveApi::surfacesList(mAttributeStack.back(),
                                                                      *mSensors));
  // give the sensors to the scene
  s->setSensors(sensors);

  // create a new Renderer
  mRenderer.reset(RendererApi::renderer(mAttributeStack.back(), mPhotonMaps));

  // create a Record
  shared_ptr<Record> record;
  record.reset(RecordApi::record(mAttributeStack.back()));

  // create a ShadingContext
  shared_ptr<ShadingContext> context;
  context.reset(ShadingApi::context(mAttributeStack.back(), mMaterials, mTextures));

  // give everything to the renderer
  mRenderer->setScene(s);
  mRenderer->setRecord(record);
  mRenderer->setShadingContext(context);

  // headless render?
  bool headless = (attr["viewer"] == std::string("false"));

  if(!headless)
  {
#if USE_QGLVIEWER
    int zero = 0;
    QApplication application(zero,0);
#endif // USE_QGLVIEWER

    RenderViewer v;

    // title the window the name of the outfile
    v.setWindowTitle(attr["record:outfile"].c_str());

    // everything to the viewer
    v.setScene(s);

    v.setRenderer(mRenderer);

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
      v.camera()->setPosition(RenderViewer::Vec(eyex,eyey,eyez));
      v.camera()->setUpVector(RenderViewer::Vec(upx,upy,upz));
      v.camera()->setViewDirection(RenderViewer::Vec(lookx,looky,lookz));

      // open the window
      v.show();

#if USE_QGLVIEWER
      application.exec();
#endif // USE_QGLVIEWER
    } // end try
    catch(...)
    {
      std::cerr << "Warning: Unable to create a window. Defaulting to headless render." << std::endl;
      Renderer::ProgressCallback callback;
      mRenderer->render(callback);
    } // end catch
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
  mMaterials->push_back(shared_ptr<Material>(m));

  // note the current material
  mAttributeStack.back()["material"] = lexical_cast<std::string>(mMaterials->size() - 1);
} // end Gotham::material)

TextureHandle Gotham
  ::texture(const std::string &filename)
{
  // create a new texture
  shared_ptr<Texture> newTex(new DeferredLoadTexture(filename.c_str()));
  mTextures->push_back(newTex);

  return mTextures->size() - 1;
} // end Gotham::texture()

TextureHandle Gotham
  ::texture(const size_t w,
            const size_t h,
            std::vector<float> &pixels)
{
  // create a new texture
  shared_ptr<Texture> newTex(new Texture(w,h, reinterpret_cast<const Spectrum*>(&pixels[0])));
  mTextures->push_back(newTex);

  return mTextures->size() - 1;
} // end Gotham::texture()

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
  surfacePrimitive(new RasterizableSurfacePrimitive(surface, lexical_cast<MaterialHandle>(attr["material"])));
} // end Gotham::sphere()

void Gotham
  ::surfacePrimitive(SurfacePrimitive *prim)
{
  // name the primitive
  prim->setName(mAttributeStack.back()["name"]);

  shared_ptr<SurfacePrimitive> surfacePrim(prim);
  shared_ptr<Primitive> plainPrim = static_pointer_cast<Primitive,SurfacePrimitive>(surfacePrim);

  // add to aggregate list of all Primitives
  mPrimitives->push_back(plainPrim);

  // add to list of all SurfacePrimitives
  mSurfaces->push_back(surfacePrim);

  const Material &m = *(*mMaterials)[surfacePrim->getMaterial()];
  if(m.isEmitter())
  {
    mEmitters->push_back(surfacePrim);
  } // end if

  if(m.isSensor())
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
  surfacePrimitive(new RasterizableSurfacePrimitive(surface, lexical_cast<MaterialHandle>(attr["material"])));
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
  surfacePrimitive(new RasterizableSurfacePrimitive(surface, lexical_cast<MaterialHandle>(attr["material"])));
} // end Gotham::mesh()

void Gotham
  ::mesh(std::vector<float> &vertices,
         std::vector<float> &parametrics,
         std::vector<float> &normals,
         std::vector<unsigned int> &faces)
{
  // do we need to reverse the winding of vertices?
  bool reverse = mAttributeStack.back()["orientation"] == "inside";

  std::vector<Point> points;
  std::vector<ParametricCoordinates> parms;
  std::vector<Normal> norms;
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
      i != normals.size();
      i += 3)
  {
    norms.push_back(Normal(normals[i], normals[i+1], normals[i+2]));

    // transform by the matrix on the top of the stack
    norms.back() = mMatrixStack.back()(norms.back());

    //// XXX do we need to flip these?
    //if(reverse)
    //{
    //  norms.back() = -norms.back();
    //} // end if
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
    mesh = new RasterizableMesh<Mesh>(points, parms, norms, triangles);
  } // end if
  else
  {
    mesh = new RasterizableMesh<SmallMesh>(points, parms, norms, triangles);
  } // end else

  shared_ptr<Surface> surface(mesh);

  AttributeMap &attr = mAttributeStack.back();
  surfacePrimitive(new RasterizableSurfacePrimitive(surface, lexical_cast<MaterialHandle>(attr["material"])));
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

std::string Gotham
  ::getAttribute(const std::string &name) const
{
  AttributeMap::const_iterator i = mAttributeStack.back().find(name);
  if(i != mAttributeStack.back().end())
  {
    return i->second;
  } // end if

  return "";
} // end Gotham::getAttribute()

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
  ::parseMesh(const std::string &line)
{
  std::vector<float> points;
  std::vector<float> uvs;
  std::vector<unsigned int> faces;

  rule<phrase_scanner_t> pointsRule  = (ch_p('(') | ch_p('[')) >> real_p[push_back_a(points)]  >> *(',' >> real_p[push_back_a(points)])  >> (ch_p(')') | ch_p(']'));
  rule<phrase_scanner_t> uvsRule = (ch_p('(') | ch_p('[')) >> real_p[push_back_a(uvs)] >> *(',' >> real_p[push_back_a(uvs)]) >> (ch_p(')') | ch_p(']'));
  rule<phrase_scanner_t> facesRule = (ch_p('(') | ch_p('[')) >> uint_p[push_back_a(faces)] >> *(',' >> uint_p[push_back_a(faces)]) >> (ch_p(')') | ch_p(']'));

  bool result = parse(line.c_str(),
    // begin grammar
    (
      // points, uvs, and faces
      str_p("Mesh") >> '(' >> pointsRule >> ',' >> uvsRule >> ',' >> facesRule >> ')' >> end_p
      |
      // points and faces
      str_p("Mesh") >> '(' >> pointsRule >> ',' >> facesRule >> ')' >> end_p
    )
    ,
    // end grammar
    
    space_p).full;

  if(result)
  {
    // validate points
    if((points.size() % 3) != 0)
    {
      // XXX report error
      throw;
    } // end if

    if(uvs.size() && ((uvs.size() % 2) != 0))
    {
      // XXX report error
      throw;
    } // end if

    // validate faces
    if((faces.size() % 3) != 0)
    {
      // XXX report error
      throw;
    } // end if

    for(size_t i = 0; i != faces.size(); ++i)
    {
      if(faces[i] >= points.size() / 3)
      {
        // face i refers to non-vertex!
        // XXX report error
        throw;
      } // end if
    } // end for i

    // instantiate the mesh
    if(points.size() && uvs.size() && faces.size())
    {
      mesh(points, uvs, faces);
    } // end if
    else if(points.size() && faces.size())
    {
      mesh(points, faces);
    } // end else
  } // end if

  return result;
} // end Gotham::parseMesh()

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
      str_p("Photons") >> '(' >> pointsRule >> ',' >> vectorsRule >> ',' >> powersRule >> ')' >> end_p
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
  else if(parseMesh(line))
  {
    result = true;
  } // end else if

  return result;
} // end Gotham::parseLine()

