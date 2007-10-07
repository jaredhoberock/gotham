/*! \file exportPrimitives.h
 *  \author Jared Hoberock
 *  \brief This file contains DLL exports and imports
 *         for classes used by shaders.
 *  \note This nonsense is unnecessary on Linux.
 */

#ifndef EXPORT_PRIMITIVES_H
#define EXPORT_PRIMITIVES_H

#ifdef WIN32
#ifdef IMPORTDLL
#define DLLAPI __declspec(dllimport)
#else
#define DLLAPI __declspec(dllexport)
#endif // IMPORTDLL

class DLLAPI Material;
class DLLAPI Surface;
#include <boost/shared_ptr.hpp>
class DLLAPI boost::detail::shared_count;
template class DLLAPI boost::shared_ptr<Surface>;
template class DLLAPI boost::shared_ptr<Material>;

class DLLAPI Primitive;
class DLLAPI SurfacePrimitive;

#endif // WIN32

#endif // EXPORT_PRIMITIVES_H

