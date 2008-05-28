/*! \file CommonViewer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a QGLViewer class
 *         containing common operations.
 */

#ifndef COMMON_VIEWER_H
#define COMMON_VIEWER_H

#ifdef WIN32
#define NOMINMAX
#define WINDOWS_LEAN_AND_MEAN
#endif // WIN32

#include <GL/glew.h>
#include <gl++/texture/Texture.h>
#include <gl++/shader/Shader.h>
#include <gl++/program/Program.h>

template<typename Parent,
         typename KeyEventType,
         typename StringType,
         typename VectorType>
  class CommonViewer
    : public Parent
{
  public:
    /*! \typedef KeyEvent
     *  \brief Make the type of KeyEvent available to children classes.
     */
    typedef KeyEventType KeyEvent;

    /*! \typedef StringType
     *  \brief Make the type of StringType available to child classes.
     */
    typedef StringType String;

    /*! \typedef VectorType
     *  \brief Make the type of VectorType available to child classes.
     */
    typedef VectorType Vec;

    /*! This method displays the given texture as a full screen quad.
     *  \param t The Texture of interest.
     */
    inline void drawTexture(const glpp::Texture &t) const;

    /*! This method displays the given texture as a full screen quad.
     *  \param t The Texture of interest.
     *  \param p The Program to use to draw the Texture.
     */
    inline void drawTexture(const glpp::Texture &t, const Program &p) const;

    inline virtual void init(void);
    inline virtual void keyPressEvent(KeyEvent *e);

    /*! This method prompts the user for a filename.
     *  \param prompt The prompt to display to the user.
     *  \param path The path to begin.
     *  \param desc A string description of files to filter.
     *  \return The complete path to the file as a string.
     */
    inline std::string getOpenFileName(const char *prompt,
                                       const char *path,
                                       const char *desc);

    /*! This method displays the given string on the screen.
     *  \param message The message to display.
     *  \param int delay
     */
    inline void displayMessage(const std::string &message,
                               int delay = 2000);

  protected:
    /*! This method reloads this CommonViewer's shaders.
     */
    inline virtual void reloadShaders(void);

    /*! Pass-through fragment shader which applies scale and gamma.
     */
    Shader mPassthroughGammaShader;

    /*! Pass-through program which applies scale and gamma.
     */
    Program mPassthroughGammaProgram;

    /*! Fragment shader to visualize a 2D Rect.
     */
    Shader mTexture2DRectShader;

    /*! Program to visualize a 2D Rect.
     */
    Program mTexture2DRectProgram;

    /*! Fragment shader to visualize a 2D Array.
     */
    Shader mTexture2DArrayShader;

    /*! Program to visualize a 2D Array.
     */
    Program mTexture2DArrayProgram;

    /*! Fragment shader to visualize a 2D Rect by normalizing by its alpha.
     */
    Shader mTexture2DRectNormalizeShader;

    /*! Program to visualize a 2D Rect by normalizing by its alpha.
     */
    Program mTexture2DRectNormalizeProgram;

    /*! Fragment shader to tonemap the visualization of a 2D Rect.
     */
    Shader mTexture2DRectTonemapShader;

    /*! Program to tonemap the visualization of a 2D Rect.
     */
    Program mTexture2DRectTonemapProgram;

    /*! Fragment shader to gamma-correct a 2D Rect.
     */
    Shader mTexture2DRectGammaShader;

    /*! Program to gamma-correct the visualization of a 2D Rect.
     */
    Program mTexture2DRectGammaProgram;

    // for compatibility with Qt
    static const unsigned int SHIFT_MODIFIER   = 0x02000000;
    static const unsigned int CONTROL_MODIFIER = 0x04000000;
    static const unsigned int ALT_MODIFIER     = 0x08000000;
}; // end CommonViewer

#include "CommonViewer.inl"

#endif // COMMON_VIEWER_H

