/*! \file MaterialViewer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a viewer class
 *         for looking at Materials on the unit square.
 */

#ifndef MATERIAL_VIEWER_H
#define MATERIAL_VIEWER_H

#include <GL/glew.h>
#include <QGLViewer/qglviewer.h>
#include <QtGui/QKeyEvent>
#include <commonviewer/CommonViewer.h>

#include "Material.h"

class MaterialViewer
  : public CommonViewer<QGLViewer, QKeyEvent>
{
  public:
    typedef CommonViewer<QGLViewer,QKeyEvent> Parent;

    inline virtual void init(void);
    inline virtual void draw(void);
    inline void setMaterial(Material *m);

  protected:
    inline void initTexture(void);
    Texture mTexture;
    Material *mMaterial;
    float3 mLightPosition;
}; // end MaterialViewer

#include "MaterialViewer.inl"

#endif // MATERIAL_VIEWER_H

