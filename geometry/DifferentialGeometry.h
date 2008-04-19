/*! \file DifferentialGeometry.h
 *  \author Jared Hoberock
 *  \brief This file instantiates DifferentialGeometry
 *         to use as a class on the CPU.
 */

#pragma once

#include "DifferentialGeometryBase.h"

#include <gpcpu/Vector.h>
#include "Point.h"
#include "Vector.h"
#include "Normal.h"
#include "ParametricCoordinates.h"

typedef DifferentialGeometryBase<Point,Vector,ParametricCoordinates,Normal> DifferentialGeometry;


