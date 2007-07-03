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

    /*! This method sets the current Material.
     *  \param name The name of the Material.
     */
    void material(const char *name);

  private:
    /*! The matrix stack.
     */
    std::vector<Matrix> mMatrixStack;
}; // end Gotham

#include "Gotham.inl"

#endif // GOTHAM_H

