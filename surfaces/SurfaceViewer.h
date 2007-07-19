/*! \file SurfaceViewer.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a viewer class for Surfaces.
 */

#ifndef SURFACE_VIEWER_H
#define SURFACE_VIEWER_H


#include <GL/glew.h>
#include <QGLViewer/qglviewer.h>
#include <QtGui/QKeyEvent>
#include <mesh/TriangleMeshViewer.h>

#include "Mesh.h"

typedef TriangleMeshViewer<QGLViewer, QKeyEvent, Point, ParametricCoordinates, Normal> SurfaceViewer;

#endif // SURFACE_VIEWER_H

