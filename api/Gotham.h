/*! \file Gotham.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to the Gotham API.
 */

#ifndef GOTHAM_H
#define GOTHAM_H 

#include <vector>
#include <list>
#include "../geometry/Transform.h"
#include "../primitives/SurfacePrimitive.h"
#include "../primitives/PrimitiveList.h"
#include "../primitives/SurfacePrimitiveList.h"
#include "../shading/MaterialList.h"
#include "../renderers/Renderer.h"
#include "../records/PhotonMap.h"
#include <boost/shared_ptr.hpp>
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include <map>

class Gotham
{
  public:
    /*! \typedef AttributeMap
     *  \brief Shorthand.
     */
    typedef std::map<std::string, std::string> AttributeMap;

    /*! \typedef PhotonMaps
     *  \brief Shorthand.
     *  XXX This probably doesn't belong here.
     */
    typedef std::map<std::string, boost::shared_ptr<PhotonMap> > PhotonMaps;

    /*! Null constructor calls init().
     */
    Gotham(void);

    /*! Null destructor does nothing.
     */
    virtual ~Gotham(void);

    /*! This method sets the initial graphics state.
     */
    virtual void init(void);

    /*! This method pushes a copy of the current Matrix
     *  to the top of the matrix stack.
     */
    void pushMatrix(void);

    /*! This method pops the top of the Matrix stack.
     */
    void popMatrix(void);

    /*! This method transforms the top of the Matrix stack
     *  by the given translation.
     *  \param tx The x coordinate of the translation vector.
     *  \param ty The y coordinate of the translation vector.
     *  \param tz The z coordinate of the translation vector.
     */
    void translate(const float tx, const float ty, const float tz);

    /*! This method transforms the top of the Matrix stack
     *  by the given rotation.
     *  \param degrees The angle of rotation; in degrees.
     *  \param rx The x coordinate of the axis of rotation.
     *  \param ry The y coordinate of the axis of rotation.
     *  \param rz The z coordinate of the axis of rotation.
     */
    void rotate(const float degrees, const float rx, const float ry, const float rz);

    /*! This method scales the top of the Matrix stack
     *  by the given scale.
     *  \param sx The scale in the x dimension.
     *  \param sy The scale in the y dimension.
     *  \param sz The scale in the z dimension.
     */
    void scale(const float sx, const float sy, const float sz);

    /*! This method multiplies the top of the Matrix stack
     *  by the given Matrix.
     *  \param m A row-major order Matrix.
     */
    void multMatrix(const std::vector<float> &m);

    /*! This method loads the given Matrix into the top of the
     *  Matrix stack.
     *  \param m The row-order Matrix to load.
     */
    void loadMatrix(const std::vector<float> &m);

    /*! This method returns the top of the Matrix stack.
     *  \param m The top of the Matrix stack is returned here.
     */
    void getMatrix(std::vector<float> &m);

    /*! This method starts a render.
     */
    virtual void render(void);

    /*! This method sets the given Material
     *  as the current Material.
     *  \param m The Material.
     */
    virtual void material(Material *m);

    /*! This method creates a new triangle Mesh by
     *  transforming the given geometry against the current
     *  Matrix.
     *  \param vertices A list of triangle vertices.
     *  \param triangles A list of vertex index triples.
     */
    virtual void mesh(std::vector<float> &vertices,
                      std::vector<unsigned int> &triangles);

    /*! This method creates a new triangle Mesh by
     *  transforming the given geometry against the current
     *  Matrix.
     *  \param vertices A list of triangle vertex positions.
     *  \param parametrics A list of parametric triangle vertex positions.
     *  \param triangles A list of vertex index triples.
     */
    virtual void mesh(std::vector<float> &vertices,
                      std::vector<float> &parametrics,
                      std::vector<unsigned int> &triangles);

    /*! This method creates a new Sphere.
     *  \param cx The x-coordinate of the center of the Sphere.
     *  \param cy The y-coordinate of the center of the Sphere.
     *  \param cz The z-coordinate of the center of the Sphere.
     *  \param radius The radius of the Sphere.
     */
    virtual void sphere(const float cx,
                        const float cy,
                        const float cz,
                        const float radius);

    /*! This method creates a new PhotonMap.
     *  \param positions A list of Photon positions.
     *  \param wi A list of Photon incoming directions.
     *  \param power A list of Photon powers.
     */
    virtual void photons(const std::vector<float> &positions,
                         const std::vector<float> &wi,
                         const std::vector<float> &power);

    /*! This method sets the given named attribute.
     *  \param name The name of the attribute to set.
     *  \param val The value to set to.
     */
    virtual void attribute(const std::string &name, const std::string &val);

    /*! This method pushes a copy of the current attributes to the top of
     *  the attributes stack.
     */
    void pushAttributes(void);

    /*! This method pops the top of the attributes stack.
     */
    void popAttributes(void);

    /*! This method tries to parse a single line of Gotham Python code.
     *  \param line The line to parse
     *  \return true if line could be successfully parsed; false, otherwise.
     */
    bool parseLine(const std::string &line);

  protected:
    typedef Transform Matrix;

    /*! This method loads the given Matrix into the Matrix
     *  on top of the matrix stack.
     *  \param m The Matrix to load.
     */
    void loadMatrix(const Matrix &m);

    /*! This method multiplies the top of the Matrix stack
     *  by the given Matrix.
     *  \param m The Matrix to multiply by.
     */
    void multMatrix(const Matrix &m);

    /*! This method returns an AttributeMap filled with default
     *  attributes.
     *  \param attr A set of default attributes is returned here.
     */
    void getDefaultAttributes(AttributeMap &attr) const;

    /*! This method accepts a new SurfacePrimitive to add to the current Scene.
     *  \param prim The new SurfacePrimitive.
     */
    void surfacePrimitive(SurfacePrimitive *prim);

    /*! This method parses a line of Gotham Python code for a photons() call.
     *  \param line The line to parse.
     *  \return true if line could be successfully parsed; false, otherwise.
     */
    bool parsePhotons(const std::string &line);

    /*! The matrix stack.
     */
    std::vector<Matrix> mMatrixStack;

    /*! The attribute stack.
     */
    std::vector<AttributeMap> mAttributeStack;

    /*! The list of Materials created for this Scene.
     */
    boost::shared_ptr<MaterialList> mMaterials;

    /*! A list of Primitives.
     */
    boost::shared_ptr<PrimitiveList> mPrimitives;

    /*! A list of SurfacePrimitives.
     */
    boost::shared_ptr<SurfacePrimitiveList> mSurfaces;

    /*! A list of SurfacePrimitives whose Materials
     *  identify themselves as emitters.
     *  XXX Perhaps this should be a list of Primitives?
     */
    boost::shared_ptr<SurfacePrimitiveList> mEmitters;

    /*! A list of SurfacePrimitives whose Materials
     *  identify themselves as sensors.
     *  XXX Perhaps this should be a list of Primitives?
     */
    boost::shared_ptr<SurfacePrimitiveList> mSensors;

    /*! The Renderer.
     */
    boost::shared_ptr<Renderer> mRenderer;

    /*! PhotonMaps, indexed by human-readable name.
     */
    PhotonMaps mPhotonMaps;
}; // end Gotham

#include "Gotham.inl"

#endif // GOTHAM_H

