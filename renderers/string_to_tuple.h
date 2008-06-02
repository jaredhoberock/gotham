/*! \file tuple_to_string.h
 *  \author Jared Hoberock
 *  \brief Specialize lexical_cast for boost::tuple
 */

#pragma once

#include <boost/tuple/tuple_io.hpp>
#include <boost/lexical_cast.hpp>

namespace boost
{
    namespace detail // stream wrapper for handling lexical conversions
    {
        template<typename T0, typename T1, typename Source>
        class lexical_stream<tuple<T0,T1>, Source>
        {
        public:
            typedef tuple<T0,T1> Target;
        private:
            typedef typename widest_char<
                typename stream_char<Target>::type,
                typename stream_char<Source>::type>::type char_type;

        public:
            lexical_stream()
            {
                stream.unsetf(std::ios::skipws);

                if(std::numeric_limits<Target>::is_specialized)
                    stream.precision(std::numeric_limits<Target>::digits10 + 1);
                else if(std::numeric_limits<Source>::is_specialized)
                    stream.precision(std::numeric_limits<Source>::digits10 + 1);
            }
            ~lexical_stream()
            {
                #if defined(BOOST_NO_STRINGSTREAM)
                stream.freeze(false);
                #endif
            }
            bool operator<<(const Source &input)
            {
                return !(stream << tuples::set_delimiter(',') << input).fail();
            }
            template<typename InputStreamable>
            bool operator>>(InputStreamable &output)
            {
                return !is_pointer<InputStreamable>::value &&
                       stream >> output &&
                       stream.get() ==
#if defined(__GNUC__) && (__GNUC__<3) && defined(BOOST_NO_STD_WSTRING)
// GCC 2.9x lacks std::char_traits<>::eof().
// We use BOOST_NO_STD_WSTRING to filter out STLport and libstdc++-v3
// configurations, which do provide std::char_traits<>::eof().
    
                           EOF;
#else
                           std::char_traits<char_type>::eof();
#endif
            }
            bool operator>>(std::string &output)
            {
                #if defined(BOOST_NO_STRINGSTREAM)
                stream << '\0';
                #endif
                output = stream.str();
                return true;
            }
            #ifndef DISABLE_WIDE_CHAR_SUPPORT
            bool operator>>(std::wstring &output)
            {
                output = stream.str();
                return true;
            }
            #endif
        private:
            #if defined(BOOST_NO_STRINGSTREAM)
            std::strstream stream;
            #elif defined(BOOST_NO_STD_LOCALE)
            std::stringstream stream;
            #else
            std::basic_stringstream<char_type> stream;
            #endif
        };
    }

}; // end boost

