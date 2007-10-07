/*! \file exportShading.h
 *  \author Jared Hoberock
 *  \brief This file contains DLL exports and imports
 *         for classes used by shaders.
 *  \note  This nonsense is unneccessary on Linux.
 */

#ifndef EXPORT_SHADING_H
#define EXPORT_SHADING_H

#ifdef WIN32
#ifdef IMPORTDLL
#define DLLAPI __declspec(dllimport)
#else
#define DLLAPI __declspec(dllexport)
#endif // IMPORTDLL

class DLLAPI Vector;
class DLLAPI Point;
class DLLAPI Spectrum;

namespace gotham
{
  float DLLAPI noise(float x, float y, float z);
} // end gotham

class DLLAPI Material;
class DLLAPI FunctionAllocator;
class DLLAPI Fresnel;
class DLLAPI FresnelDielectric;
class DLLAPI FresnelConductor;
class DLLAPI ScatteringDistributionFunction;
class DLLAPI CompositeDistributionFunction;
class DLLAPI Lambertian;
class DLLAPI HemisphericalEmission;
class DLLAPI PerspectiveSensor;
class DLLAPI DeltaDistributionFunction;
class DLLAPI SpecularReflection;
class DLLAPI SpecularTransmission;
class DLLAPI PerfectGlass;
class DLLAPI ThinGlass;
class DLLAPI TransparentTransmission;
class DLLAPI PhongReflection;
class DLLAPI PhongTransmission;
class DLLAPI AshikhminShirleyReflection;

#endif // WIN32

#endif // EXPORT_SHADING_H

