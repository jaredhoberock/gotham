/*! \file RussianRoulette.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         for computing Russian roulette probabilities.
 */

#ifndef RUSSIAN_ROULETTE_H
#define RUSSIAN_ROULETTE_H

// for size_t
#include <cstdlib>

class Spectrum;
class Vector;

#include "../include/DifferentialGeometry.h"

class RussianRoulette
{
  public:
    /*! Null destructor does nothing.
     */
    inline virtual ~RussianRoulette(void){;};

    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const = 0;

    /*! This method computes whether or not a Path should be extended with
     *  another PathVertex, knowing nothing about the previous PathVertex.
     *  Alternatively, this method could be used when there is no previous vertex,
     *  such as is the case of the first PathVertex of a path.
     *  \return The probability of adding a PathVertex.
     *  \note The default implementation of this function returns 1.0f.
     */
    virtual float operator()(void) const;
}; // end RussianRoulette

// always ballin'
class AlwaysRoulette
  : public RussianRoulette
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef RussianRoulette Parent;

    /*! Null destructor does nothing.
     */
    inline virtual ~AlwaysRoulette(void){;};

    using Parent::operator();
    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;
}; // end AlwaysRoulette

class ConstantRoulette
  : public RussianRoulette
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef RussianRoulette Parent;

    ConstantRoulette(const float continueProbability = 0.3f);

    /*! Null destructor does nothing.
     */
    inline virtual ~ConstantRoulette(void){;};

    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;

    virtual float operator()(void) const;

    float getContinueProbability(void) const;

  protected:
    float mContinueProbability;
}; // end ConstantRoulette

class LuminanceRoulette
  : public RussianRoulette
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef RussianRoulette Parent;

    /*! Null destructor does nothing.
     */
    inline virtual ~LuminanceRoulette(void){;};

    using Parent::operator();
    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;
}; // end LuminanceRoulette

class MaxOverSpectrumRoulette
  : public RussianRoulette
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef RussianRoulette Parent;

    /*! Null destructor does nothing.
     */
    inline virtual ~MaxOverSpectrumRoulette(void){;};

    using Parent::operator();
    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;
}; // end MaxOverSpectrumRoulette

class OnlyAfterDeltaRoulette
  : public RussianRoulette
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef RussianRoulette Parent;

    /*! Null destructor does nothing.
     */
    inline virtual ~OnlyAfterDeltaRoulette(void){;};

    using Parent::operator();
    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;
}; // end OnlyAfterDeltaRoulette

class ConstantAndAlwaysAfterDeltaRoulette
  : public ConstantRoulette
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ConstantRoulette Parent;

    /*! Null destructor does nothing.
     */
    inline virtual ~ConstantAndAlwaysAfterDeltaRoulette(void){;};

    ConstantAndAlwaysAfterDeltaRoulette(const float continueProbability = 0.3f);

    using Parent::operator();
    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;
}; // end ConstantAndAlwaysAfterDeltaRoulette

class KelemenRoulette
  : public ConstantRoulette
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ConstantRoulette Parent;

    KelemenRoulette(const float continueProbability = 0.3f);

    /*! Null destructor does nothing.
     */
    inline virtual ~KelemenRoulette(void){;};

    using Parent::operator();
    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;
}; // end KelemenRoulette

/*! \class VeachRoulette
 *  \brief This RussianRoulette class implements the Russian roulette
 *         strategy described by Veach in section 10.3.3 of His thesis.
 */
class VeachRoulette
  : public RussianRoulette
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef RussianRoulette Parent;

    /*! Constructor accepts a minimal subpath length (which must be at least 1).
     *  \param minimalSubpathLength Sets mMinimalSubpathLength.
     */
    VeachRoulette(const size_t minimalSubpathLength = 3);

    /*! Null destructor does nothing.
     */
    inline virtual ~VeachRoulette(void){;};

    using Parent::operator();

    /*! This method returns 1.0f if i is less than mMinimalSubpathLength, or fromDelta is true;
     *  otherwise, it returns the same as MaxOverSpectrumRoulette::operator()().
     *  \param i The index of the proposed new PathVertex.
     *  \param f The throughput of the proposed bounce.
     *  \param dg The DifferentialGeometry of the prior PathVertex.
     *  \param w The proposed bounce direction from the prior PathVertex.
     *  \param pdf The value of the solid angle pdf at w.
     *  \param fromDelta Whether or not w was chosen from a delta distribution.
     *  \return The probability of accepting the proposed Path extension.
     */
    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;

  protected:
    size_t mMinimalSubpathLength;
}; // end VeachRoulette

/*! \class ModifiedKelemenRoulette
 *  \brief This RussianRoulette class modifies KelemenRoulette to optionally
 *         terminate Paths before they are even started.
 */
class ModifiedKelemenRoulette
  : public RussianRoulette
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef RussianRoulette Parent;

    /*! Constructor accepts a constant probability for beginning
     *  a Path from nothing.
     */
    ModifiedKelemenRoulette(const float beginProbability);

    /*! Null destructor does nothing.
     */
    inline virtual ~ModifiedKelemenRoulette(void){;};

    /*! This method returns a probability for begining a new Path
     *  from scratch.
     *  \return mBeginProbability
     */
    virtual float operator()(void) const;

    /*! This method returns 1.0f if fromDelta is true;
     *  otherwise, it returns the same as MaxOverSpectrumRoulette::operator()().
     *  \param i The index of the proposed new PathVertex.
     *           This parameter is ignored.
     *  \param f The throughput of the proposed bounce.
     *  \param dg The DifferentialGeometry of the prior PathVertex.
     *  \param w The proposed bounce direction from the prior PathVertex.
     *  \param pdf The value of the solid angle pdf at w.
     *  \param fromDelta Whether or not w was chosen from a delta distribution.
     *  \return The probability of accepting the proposed Path extension.
     */
    virtual float operator()(const unsigned int i,
                             const Spectrum &f,
                             const DifferentialGeometry &dg,
                             const Vector &w,
                             const float &pdf,
                             const bool fromDelta) const;

  protected:
    float mBeginProbability;
}; // end ModifiedKelemenRoulette

#endif // RUSSIAN_ROULETTE_H

