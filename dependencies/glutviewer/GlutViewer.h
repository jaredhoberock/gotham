/*!	\file	GlutViewer.h
 *	\author Yuntao Jia, Jared Hoberock
 *	\brief	A class wrapper for glut inspired by
 *	        Gilles Debunne's excellent QGLViewer library.
 *          More details here: http://artis.imag.fr/Members/Gilles.Debunne/QGLViewer/index.html.
 */

#ifndef GLUT_VIEWER_H
#define GLUT_VIEWER_H

#include <string>
#include "KeyEvent.h"
#include <gpcpu/Vector.h>

class GlutViewer
{
  public:	
    class Camera
    {
      public:
        inline Camera(void);
        inline void setPosition(const gpcpu::float3 &pos);
        inline gpcpu::float3 position(void) const;
        inline void setUpVector(const gpcpu::float3 &up);
        inline gpcpu::float3 upVector(void) const;
        inline void setViewDirection(const gpcpu::float3 &direction);
        inline gpcpu::float3 viewDirection(void) const;
        inline void setFieldOfView(float fov);
        inline float fieldOfView(void) const;
        inline void setAspectRatio(float aspect);
        inline float aspectRatio(void) const;
        inline float zNear(void) const;

      protected:
        gpcpu::float3 mPosition;
        gpcpu::float3 mUpVector;
        gpcpu::float3 mViewDirection;
        float mFieldOfView;
        float mAspectRatio;
        float mZNear;
    }; // end Camera

    /* This method shows the window on screen and starts
     * the render loop.
     */
    inline virtual void show(void);

    /*! Singleton access */
    inline static GlutViewer* getInstance(void);
    
    inline GlutViewer(void);
    inline virtual ~GlutViewer(void);
    
    /* static call back function for glut */    
    inline static void displayFunc(void);	
    inline static void idleFunc(void);
    inline static void timerFunc(int value);
    inline static void keyFunc(unsigned char key, int x, int y);
    inline static void mouseFunc(int button, int state, int x, int y);
    inline static void motionFunc(int x, int y);
    inline static void reshapeFunc(int w, int h);

    /*! This method is called upon a glut resize event.
     *  \param w The new width of the window.
     *  \param h The new height of the window.
     */
    inline virtual void resizeGL(int w, int h);

    /*! This method calls glutPostRedisplay().
     */
    inline virtual void updateGL(void);

    /*! This method returns the width of this GlutViewer's window.
     *  \return mWidth.
     */
    inline int width(void) const;

    /*! This method returns the height of this GlutViewer's window.
     *  \return mHeight.
     */
    inline int height(void) const;

    /*! This method starts the animation.
     */
    inline virtual void startAnimation(void);

    /*! This method stops the animation.
     */
    inline virtual void stopAnimation(void);

    /*! This method returns true when this GlutViewer is animating.
     *  \return mIsAnimating
     */
    inline bool animationIsStarted(void) const;

    // This method displays a message in the lower left hand corner of the screen.
    inline virtual void displayMessage(const std::string &message, int delay = 2000);

    // This method does nothing, but is included to conform to the QGLViewer interface.
    inline virtual void setStateFileName(const std::string &filename);

    // This method does nothing, but is included to conform to the QGLViewer interface.
    inline virtual void setSceneBoundingBox(const gpcpu::float3 &min,
                                            const gpcpu::float3 &max);
    
    // This method returns a pointer to mCamera.
    // \return &mCamera
    inline Camera *camera(void) const;

    // This method sets the dimensions of this GlutViewer
    inline void resize(const int w, const int h);

    // This method sets the window title
    inline void setWindowTitle(const std::string &title);

    // This method sets the animation period.
    inline void setAnimationPeriod(int period);

  protected:
    /* interface for children classes */
    inline virtual void init(void);
    inline virtual void draw(void);
    inline virtual void animate(void);
    inline virtual void beginDraw(void);
    inline virtual void endDraw(void);
    inline virtual void keyPressEvent(KeyEvent *e);

    inline virtual void drawText(int id, const std::string &text);

  private:
    inline virtual void render(void);

    // this method initializes glut
    inline void initializeGlut(void);

    // this method initialize GL
    inline void initializeGL(void);
    
    // internal method, including message and fps
    inline void drawText(void);

    /* instance */
    static GlutViewer *viewer;	

    // for window
    int mWidth;
    int mHeight;
	
    // for message display
    static const int MESSAGE_INDENT = 20;
    int m_nMsgShiftX;
    int m_nMsgShiftY;
    char  m_Msg[256];
    
    int mMessageDieTime;

    bool mIsAnimating;

    mutable Camera mCamera;

    // the title to display on the window
    std::string mWindowTitle;

    // The animation period
    int mAnimationPeriod;
}; // end GlutViewer

#include "GlutViewer.inl"

#endif // GLUT_VIEWER_H

