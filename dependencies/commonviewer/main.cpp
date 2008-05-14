/*! \file main.cpp
 *  \author Jared Hoberock
 *  \brief Main file for commonviewer application.
 */

#include <GL/glew.h>
#include <QGLViewer/qglviewer.h>
#include <viewermain/main.h>
#include "CommonViewer.h"

#include <QtGui/QKeyEvent>

int main(int argc, char **argv)
{
  return viewerMain<CommonViewer<QGLViewer, QKeyEvent> >(argc, argv);
} // end main()

