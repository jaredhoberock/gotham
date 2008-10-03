/*! \file exportApi.h
 *  \author Jared Hoberock
 *  \brief Implementation of exportApi function.
 */

#include "exportApi.h"
#include "Gotham.h"
#include "../include/detail/TextureHandle.h"
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
using namespace boost::python;

// wrapper for Gotham::material()
// see http://www.boost.org/libs/python/doc/v2/faq.html#ownership
MaterialHandle Gotham_material_ptr(Gotham &g, std::auto_ptr<Material> m)
{
  MaterialHandle result = g.material(m.get());
  m.release();

  return result;
} // end Gotham_material_ptr()

void Gotham_material_handle(Gotham &g, const MaterialHandle m)
{
  g.material(m);
} // end Gotham_material_handle()

// tell boost which multMatrix we mean
typedef void (Gotham::*multMatrix_vector)(const std::vector<float>&);

// tell boost which loadMatrix we mean
typedef void (Gotham::*loadMatrix_vector)(const std::vector<float>&);

// tell boost which getMatrix we mean
typedef void (Gotham::*getMatrix_vector)(const std::vector<float>&);

// deal with overloaded Gotham::mesh
void (Gotham::*mesh2)(std::vector<float>&,
                      std::vector<unsigned int>&)
  = &Gotham::mesh;

void (Gotham::*mesh3)(std::vector<float>&,
                      std::vector<float>&,
                      std::vector<unsigned int> &)
  = &Gotham::mesh;

void (Gotham::*mesh4)(std::vector<float>&,
                      std::vector<float>&,
                      std::vector<float>&,
                      std::vector<unsigned int> &)
  = &Gotham::mesh;

// deal with overloaded Gotham::texture
TextureHandle (Gotham::*texture1)(const std::string &)
  = &Gotham::texture;

TextureHandle (Gotham::*texture3)(const size_t,
                                  const size_t,
                                  std::vector<float> &)
  = &Gotham::texture;

void exportGotham(void)
{
  class_<Gotham>("Gotham")
    .def(init<const Gotham&>())
    .def("pushMatrix", &Gotham::pushMatrix)
    .def("popMatrix", &Gotham::popMatrix)
    .def("translate", &Gotham::translate)
    .def("rotate", &Gotham::rotate)
    .def("scale", &Gotham::scale)
    .def("multMatrix", multMatrix_vector(&Gotham::multMatrix))
    .def("loadMatrix", loadMatrix_vector(&Gotham::loadMatrix))
    .def("getMatrix", getMatrix_vector(&Gotham::getMatrix))
    .def("mesh", mesh2)
    .def("mesh", mesh3)
    .def("mesh", mesh4)
    .def("sphere", &Gotham::sphere)
    .def("render", &Gotham::render)
    .def("material", Gotham_material_ptr)
    .def("material", Gotham_material_handle)
    .def("texture", texture1)
    .def("texture", texture3)
    .def("attribute", &Gotham::attribute)
    .def("getAttribute", &Gotham::getAttribute)
    .def("pushAttributes", &Gotham::pushAttributes)
    .def("popAttributes", &Gotham::popAttributes)
    .def("photons", &Gotham::photons)
    .def("parseLine", &Gotham::parseLine)
    ;
} // end exportGotham()

void exportTextureParameter(void)
{
  class_<TextureParameter>("TextureParameter")
    .def_readwrite("mHandle", &TextureParameter::mHandle)
    .def_readwrite("mAlias", &TextureParameter::mAlias)
    ;
} // end exportTextureParameter()

void exportMaterial(void)
{
  class_<Material, std::auto_ptr<Material> >("Material")
    ;
} // end exportMaterial()

void exportVectorFloat(void)
{
  class_<std::vector<float> >("vector_float")
    .def(vector_indexing_suite<std::vector<float> >())
    ;
} // end exportVectorFloat()

void exportVectorUint(void)
{
  class_<std::vector<unsigned int> >("vector_uint")
    .def(vector_indexing_suite<std::vector<unsigned int> >())
    ;
} // end exportVectorUint()

void exportApi(void)
{
  exportGotham();
  exportTextureParameter();
  exportMaterial();
  exportVectorFloat();
  exportVectorUint();
} // end exportApi()

BOOST_PYTHON_MODULE(libgotham)
{
  exportApi();
} // end BOOST_PYTHON_MODULE()

