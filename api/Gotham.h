/*! \file Gotham.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to the Gotham API.
 */

#ifndef GOTHAM_H
#define GOTHAM_H

#include <vector>
#include <list>
#include "../geometry/Transform.h"
#include "../shading/Material.h"
#include "../primitives/SurfacePrimitive.h"
#include "../primitives/PrimitiveList.h"
#include "../primitives/SurfacePrimitiveList.h"
#include "../renderers/Renderer.h"
#include <boost/shared_ptr.hpp>

class Gotham
{
  public:
    /*! Null constructor calls init().
     */
    Gotham(void);

    /*! Null destructor does nothing.
     */
    virtual ~Gotham(void);

    /*! This method sets the initial graphics state.
     */
    void init(void);

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
     *  \param width The width of the image to render.
     *  \param height The height of the image to render.
     */
    void render(const unsigned int width,
                const unsigned int height);

    /*! This method sets the given Material
     *  as the current Material.
     *  \param m The Material.
     */
    void material(Material *m);

    /*! This method creates a new triangle Mesh by
     *  transforming the given geometry against the current
     *  Matrix.
     *  \param vertices A list of triangle vertices.
     *  \param triangles A list of vertex index triples.
     */
    void mesh(std::vector<float> &vertices,
              std::vector<unsigned int> &triangles);

  private:
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

    /*! The matrix stack.
     */
    std::vector<Matrix> mMatrixStack;

    /*! The current material.
     */
    boost::shared_ptr<Material> mCurrentMaterial;

    /*! A list of SurfacePrimitives.
     */
    boost::shared_ptr<PrimitiveList<> > mPrimitives;

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
}; // end Gotham

#include "Gotham.inl"

#endif // GOTHAM_H

