/*! \file GpuFilterFilm.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a GpuFilm class
 *         who performs a pixel reconstruction filter
 *         while depositing.
 */

#ifndef GPU_FILTER_FILM_H
#define GPU_FILTER_FILM_H

#include "GpuFilm.h"
#include <gl++/shader/Shader.h>
#include <gl++/program/Program.h>

template<typename ParentFilmType>
  class GpuFilterFilm
    : public GpuFilm<ParentFilmType>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef GpuFilm<ParentFilmType> Parent;

    /*! This method calls the Parent and creates this GpuFilterFilm's
     *  shaders and program.
     *  \note This method assumes an OpenGL context exists.
     */
    inline virtual void init(void);

  //protected:
    /*! This method reloads and relinks the shaders.
     */
    inline virtual void reloadShaders(void);

    /*! This method renders all pending deposits to mTexture.
     */
    virtual void renderPendingDeposits(void);

    // A Vertex shader to pass image locations to the geometry shader
    Shader mDepositVertexShader;

    // A Geometry shader to expand sample points to filter support regions
    Shader mDepositGeometryShader;

    // A Fragment shader to evaluate a filter function under each pixel in a
    // sample's support
    Shader mDepositFragmentShader;

    // A Program to deposit sample points across the image
    Program mDepositProgram;
}; // end GpuFilterFilm

#include "GpuFilterFilm.inl"

#endif // GPU_FILTER_FILM_H

