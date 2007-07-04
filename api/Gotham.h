/*! \file Gotham.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to the Gotham API.
 */

#ifndef GOTHAM_H
#define GOTHAM_H

#include <vector>
#include <gpcpu/floatmxn.h>
#include "../shading/Material.h"

class Gotham
{
  public:
    typedef float4x4 Matrix;

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

    /*! This method loads the given Matrix into the Matrix
     *  on top of the matrix stack.
     *  \param m The Matrix to load.
     */
    void loadMatrix(const Matrix &m);

    /*! This method starts a render.
     */
    void render(void);

    /*! This method loads sets the named Material
     *  as the current Material.  If the Material can not be found,
     *  the current Material is not altered.
     *  \param name The name of the Material.
     *  \return true if the Material could be successfully set;
     *          false, otherwise.
     */
    bool material(const char *name);

    /*! This method loads a Material from a shared library.
     *  \param path The path to the shared library.
     *  \return A pointer to the newly created Material if
     *          the shared library could be successfully loaded;
     *          false, otherwise.
     */
    static Material *loadMaterial(const char *path);

  private:
    /*! The matrix stack.
     */
    std::vector<Matrix> mMatrixStack;

    /*! The current material.
     */
    Material *mCurrentMaterial;
}; // end Gotham

#include "Gotham.inl"

#endif // GOTHAM_H

