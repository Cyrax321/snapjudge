//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.11.3
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013-2023 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT

/****************************************************************************\
 * Note on documentation: The source files contain links to the online      *
 * documentation of the public API at https://json.nlohmann.me. This URL    *
 * contains the most recent documentation and should also be applicable to  *
 * previous versions; documentation for deprecated functions is not         *
 * removed, but marked deprecated. See "Generate documentation" section in  *
 * file docs/README.md.                                                     *
\****************************************************************************/

#ifndef INCLUDE_NLOHMANN_JSON_HPP_
#define INCLUDE_NLOHMANN_JSON_HPP_

#include <algorithm> // all_of, find, for_each
#include <cstddef> // nullptr_t, ptrdiff_t, size_t
#include <functional> // hash, less
#include <initializer_list> // initializer_list
#ifndef JSON_NO_IO
    #include <iosfwd> // istream, ostream
#endif  // JSON_NO_IO
#include <iterator> // random_access_iterator_tag
#include <memory> // unique_ptr
#include <string> // string, stoi, to_string
#include <utility> // declval, forward, move, pair, swap
#include <vector> // vector

// #include <nlohmann/adl_serializer.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.11.3
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013-2023 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <utility>

// #include <nlohmann/detail/abi_macros.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.11.3
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013-2023 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



// This file contains all macro definitions affecting or depending on the ABI

#ifndef JSON_SKIP_LIBRARY_VERSION_CHECK
    #if defined(NLOHMANN_JSON_VERSION_MAJOR) && defined(NLOHMANN_JSON_VERSION_MINOR) && defined(NLOHMANN_JSON_VERSION_PATCH)
        #if NLOHMANN_JSON_VERSION_MAJOR != 3 || NLOHMANN_JSON_VERSION_MINOR != 11 || NLOHMANN_JSON_VERSION_PATCH != 3
            #warning "Already included a different version of the library!"
        #endif
    #endif
#endif

#define NLOHMANN_JSON_VERSION_MAJOR 3   // NOLINT(modernize-macro-to-enum)
#define NLOHMANN_JSON_VERSION_MINOR 11  // NOLINT(modernize-macro-to-enum)
#define NLOHMANN_JSON_VERSION_PATCH 3   // NOLINT(modernize-macro-to-enum)

#ifndef JSON_DIAGNOSTICS
    #define JSON_DIAGNOSTICS 0
#endif

#ifndef JSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON
    #define JSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON 0
#endif

#if JSON_DIAGNOSTICS
    #define NLOHMANN_JSON_ABI_TAG_DIAGNOSTICS _diag
#else
    #define NLOHMANN_JSON_ABI_TAG_DIAGNOSTICS
#endif

#if JSON_USE_LEGACY_DISCARDED_VALUE_COMPARISON
    #define NLOHMANN_JSON_ABI_TAG_LEGACY_DISCARDED_VALUE_COMPARISON _ldvcmp
#else
    #define NLOHMANN_JSON_ABI_TAG_LEGACY_DISCARDED_VALUE_COMPARISON
#endif

#ifndef NLOHMANN_JSON_NAMESPACE_NO_VERSION
    #define NLOHMANN_JSON_NAMESPACE_NO_VERSION 0
#endif

// Construct the namespace ABI tags component
#define NLOHMANN_JSON_ABI_TAGS_CONCAT_EX(a, b) json_abi ## a ## b
#define NLOHMANN_JSON_ABI_TAGS_CONCAT(a, b) \
    NLOHMANN_JSON_ABI_TAGS_CONCAT_EX(a, b)

#define NLOHMANN_JSON_ABI_TAGS                                       \
    NLOHMANN_JSON_ABI_TAGS_CONCAT(                                   \
            NLOHMANN_JSON_ABI_TAG_DIAGNOSTICS,                       \
            NLOHMANN_JSON_ABI_TAG_LEGACY_DISCARDED_VALUE_COMPARISON)

// Construct the namespace version component
#define NLOHMANN_JSON_NAMESPACE_VERSION_CONCAT_EX(major, minor, patch) \
    _v ## major ## _ ## minor ## _ ## patch
#define NLOHMANN_JSON_NAMESPACE_VERSION_CONCAT(major, minor, patch) \
    NLOHMANN_JSON_NAMESPACE_VERSION_CONCAT_EX(major, minor, patch)

#if NLOHMANN_JSON_NAMESPACE_NO_VERSION
#define NLOHMANN_JSON_NAMESPACE_VERSION
#else
#define NLOHMANN_JSON_NAMESPACE_VERSION                                 \
    NLOHMANN_JSON_NAMESPACE_VERSION_CONCAT(NLOHMANN_JSON_VERSION_MAJOR, \
                                           NLOHMANN_JSON_VERSION_MINOR, \
                                           NLOHMANN_JSON_VERSION_PATCH)
#endif

// Combine namespace components
#define NLOHMANN_JSON_NAMESPACE_CONCAT_EX(a, b) a ## b
#define NLOHMANN_JSON_NAMESPACE_CONCAT(a, b) \
    NLOHMANN_JSON_NAMESPACE_CONCAT_EX(a, b)

#ifndef NLOHMANN_JSON_NAMESPACE
#define NLOHMANN_JSON_NAMESPACE               \
    nlohmann::NLOHMANN_JSON_NAMESPACE_CONCAT( \
            NLOHMANN_JSON_ABI_TAGS,           \
            NLOHMANN_JSON_NAMESPACE_VERSION)
#endif

#ifndef NLOHMANN_JSON_NAMESPACE_BEGIN
#define NLOHMANN_JSON_NAMESPACE_BEGIN                \
    namespace nlohmann                               \
    {                                                \
    inline namespace NLOHMANN_JSON_NAMESPACE_CONCAT( \
                NLOHMANN_JSON_ABI_TAGS,              \
                NLOHMANN_JSON_NAMESPACE_VERSION)     \
    {
#endif

#ifndef NLOHMANN_JSON_NAMESPACE_END
#define NLOHMANN_JSON_NAMESPACE_END                                     \
    }  /* namespace (inline namespace) NOLINT(readability/namespace) */ \
    }  // namespace nlohmann
#endif

// #include <nlohmann/detail/conversions/from_json.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.11.3
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013-2023 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <algorithm> // transform
#include <array> // array
#include <forward_list> // forward_list
#include <iterator> // inserter, front_inserter, end
#include <map> // map
#include <string> // string
#include <tuple> // tuple, make_tuple
#include <type_traits> // is_arithmetic, is_same, is_enum, underlying_type, is_convertible
#include <unordered_map> // unordered_map
#include <utility> // pair, declval
#include <valarray> // valarray

// #include <nlohmann/detail/exceptions.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.11.3
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013-2023 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <cstddef> // nullptr_t
#include <exception> // exception
#if JSON_DIAGNOSTICS
    #include <numeric> // accumulate
#endif
#include <stdexcept> // runtime_error
#include <string> // to_string
#include <vector> // vector

// #include <nlohmann/detail/value_t.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.11.3
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013-2023 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <array> // array
#include <cstddef> // size_t
#include <cstdint> // uint8_t
#include <string> // string

// #include <nlohmann/detail/macro_scope.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.11.3
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013-2023 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <utility> // declval, pair
// #include <nlohmann/detail/meta/detected.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.11.3
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013-2023 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



#include <type_traits>

// #include <nlohmann/detail/meta/void_t.hpp>
//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.11.3
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013-2023 Niels Lohmann <https://nlohmann.me>
// SPDX-License-Identifier: MIT



// #include <nlohmann/detail/abi_macros.hpp>


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

template<typename ...Ts> struct make_void
{
    using type = void;
};
template<typename ...Ts> using void_t = typename make_void<Ts...>::type;

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END


NLOHMANN_JSON_NAMESPACE_BEGIN
namespace detail
{

// https://en.cppreference.com/w/cpp/experimental/is_detected
struct nonesuch
{
    nonesuch() = delete;
    ~nonesuch() = delete;
    nonesuch(nonesuch const&) = delete;
    nonesuch(nonesuch const&&) = delete;
    void operator=(nonesuch const&) = delete;
    void operator=(nonesuch&&) = delete;
};

template<class Default,
         class AlwaysVoid,
         template<class...> class Op,
         class... Args>
struct detector
{
    using value_t = std::false_type;
    using type = Default;
};

template<class Default, template<class...> class Op, class... Args>
struct detector<Default, void_t<Op<Args...>>, Op, Args...>
{
    using value_t = std::true_type;
    using type = Op<Args...>;
};

template<template<class...> class Op, class... Args>
using is_detected = typename detector<nonesuch, void, Op, Args...>::value_t;

template<template<class...> class Op, class... Args>
struct is_detected_lazy : is_detected<Op, Args...> { };

template<template<class...> class Op, class... Args>
using detected_t = typename detector<nonesuch, void, Op, Args...>::type;

template<class Default, template<class...> class Op, class... Args>
using detected_or = detector<Default, void, Op, Args...>;

template<class Default, template<class...> class Op, class... Args>
using detected_or_t = typename detected_or<Default, Op, Args...>::type;

template<class Expected, template<class...> class Op, class... Args>
using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

template<class To, template<class...> class Op, class... Args>
using is_detected_convertible =
    std::is_convertible<detected_t<Op, Args...>, To>;

}  // namespace detail
NLOHMANN_JSON_NAMESPACE_END

// #include <nlohmann/thirdparty/hedley/hedley.hpp>


//     __ _____ _____ _____
//  __|  |   __|     |   | |  JSON for Modern C++
// |  |  |__   |  |  | | | |  version 3.11.3
// |_____|_____|_____|_|___|  https://github.com/nlohmann/json
//
// SPDX-FileCopyrightText: 2013-2023 Niels Lohmann <https://nlohmann.me>
// SPDX-FileCopyrightText: 2016-2021 Evan Nemerson <evan@nemerson.com>
// SPDX-License-Identifier: MIT

/* Hedley - https://nemequ.github.io/hedley
 * Created by Evan Nemerson <evan@nemerson.com>
 */

#if !defined(JSON_HEDLEY_VERSION) || (JSON_HEDLEY_VERSION < 15)
#if defined(JSON_HEDLEY_VERSION)
    #undef JSON_HEDLEY_VERSION
#endif
#define JSON_HEDLEY_VERSION 15

#if defined(JSON_HEDLEY_STRINGIFY_EX)
    #undef JSON_HEDLEY_STRINGIFY_EX
#endif
#define JSON_HEDLEY_STRINGIFY_EX(x) #x

#if defined(JSON_HEDLEY_STRINGIFY)
    #undef JSON_HEDLEY_STRINGIFY
#endif
#define JSON_HEDLEY_STRINGIFY(x) JSON_HEDLEY_STRINGIFY_EX(x)

#if defined(JSON_HEDLEY_CONCAT_EX)
    #undef JSON_HEDLEY_CONCAT_EX
#endif
#define JSON_HEDLEY_CONCAT_EX(a,b) a##b

#if defined(JSON_HEDLEY_CONCAT)
    #undef JSON_HEDLEY_CONCAT
#endif
#define JSON_HEDLEY_CONCAT(a,b) JSON_HEDLEY_CONCAT_EX(a,b)

#if defined(JSON_HEDLEY_CONCAT3_EX)
    #undef JSON_HEDLEY_CONCAT3_EX
#endif
#define JSON_HEDLEY_CONCAT3_EX(a,b,c) a##b##c

#if defined(JSON_HEDLEY_CONCAT3)
    #undef JSON_HEDLEY_CONCAT3
#endif
#define JSON_HEDLEY_CONCAT3(a,b,c) JSON_HEDLEY_CONCAT3_EX(a,b,c)

#if defined(JSON_HEDLEY_VERSION_ENCODE)
    #undef JSON_HEDLEY_VERSION_ENCODE
#endif
#define JSON_HEDLEY_VERSION_ENCODE(major,minor,revision) (((major) * 1000000) + ((minor) * 1000) + (revision))

#if defined(JSON_HEDLEY_VERSION_DECODE_MAJOR)
    #undef JSON_HEDLEY_VERSION_DECODE_MAJOR
#endif
#define JSON_HEDLEY_VERSION_DECODE_MAJOR(version) ((version) / 1000000)

#if defined(JSON_HEDLEY_VERSION_DECODE_MINOR)
    #undef JSON_HEDLEY_VERSION_DECODE_MINOR
#endif
#define JSON_HEDLEY_VERSION_DECODE_MINOR(version) (((version) % 1000000) / 1000)

#if defined(JSON_HEDLEY_VERSION_DECODE_REVISION)
    #undef JSON_HEDLEY_VERSION_DECODE_REVISION
#endif
#define JSON_HEDLEY_VERSION_DECODE_REVISION(version) ((version) % 1000)

#if defined(JSON_HEDLEY_GNUC_VERSION)
    #undef JSON_HEDLEY_GNUC_VERSION
#endif
#if defined(__GNUC__) && defined(__GNUC_PATCHLEVEL__)
    #define JSON_HEDLEY_GNUC_VERSION JSON_HEDLEY_VERSION_ENCODE(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#elif defined(__GNUC__)
    #define JSON_HEDLEY_GNUC_VERSION JSON_HEDLEY_VERSION_ENCODE(__GNUC__, __GNUC_MINOR__, 0)
#endif

#if defined(JSON_HEDLEY_GNUC_VERSION_CHECK)
    #undef JSON_HEDLEY_GNUC_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_GNUC_VERSION)
    #define JSON_HEDLEY_GNUC_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_GNUC_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_GNUC_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_MSVC_VERSION)
    #undef JSON_HEDLEY_MSVC_VERSION
#endif
#if defined(_MSC_FULL_VER) && (_MSC_FULL_VER >= 140000000) && !defined(__ICL)
    #define JSON_HEDLEY_MSVC_VERSION JSON_HEDLEY_VERSION_ENCODE(_MSC_FULL_VER / 10000000, (_MSC_FULL_VER % 10000000) / 100000, (_MSC_FULL_VER % 100000) / 100)
#elif defined(_MSC_FULL_VER) && !defined(__ICL)
    #define JSON_HEDLEY_MSVC_VERSION JSON_HEDLEY_VERSION_ENCODE(_MSC_FULL_VER / 1000000, (_MSC_FULL_VER % 1000000) / 10000, (_MSC_FULL_VER % 10000) / 10)
#elif defined(_MSC_VER) && !defined(__ICL)
    #define JSON_HEDLEY_MSVC_VERSION JSON_HEDLEY_VERSION_ENCODE(_MSC_VER / 100, _MSC_VER % 100, 0)
#endif

#if defined(JSON_HEDLEY_MSVC_VERSION_CHECK)
    #undef JSON_HEDLEY_MSVC_VERSION_CHECK
#endif
#if !defined(JSON_HEDLEY_MSVC_VERSION)
    #define JSON_HEDLEY_MSVC_VERSION_CHECK(major,minor,patch) (0)
#elif defined(_MSC_VER) && (_MSC_VER >= 1400)
    #define JSON_HEDLEY_MSVC_VERSION_CHECK(major,minor,patch) (_MSC_FULL_VER >= ((major * 10000000) + (minor * 100000) + (patch)))
#elif defined(_MSC_VER) && (_MSC_VER >= 1200)
    #define JSON_HEDLEY_MSVC_VERSION_CHECK(major,minor,patch) (_MSC_FULL_VER >= ((major * 1000000) + (minor * 10000) + (patch)))
#else
    #define JSON_HEDLEY_MSVC_VERSION_CHECK(major,minor,patch) (_MSC_VER >= ((major * 100) + (minor)))
#endif

#if defined(JSON_HEDLEY_INTEL_VERSION)
    #undef JSON_HEDLEY_INTEL_VERSION
#endif
#if defined(__INTEL_COMPILER) && defined(__INTEL_COMPILER_UPDATE) && !defined(__ICL)
    #define JSON_HEDLEY_INTEL_VERSION JSON_HEDLEY_VERSION_ENCODE(__INTEL_COMPILER / 100, __INTEL_COMPILER % 100, __INTEL_COMPILER_UPDATE)
#elif defined(__INTEL_COMPILER) && !defined(__ICL)
    #define JSON_HEDLEY_INTEL_VERSION JSON_HEDLEY_VERSION_ENCODE(__INTEL_COMPILER / 100, __INTEL_COMPILER % 100, 0)
#endif

#if defined(JSON_HEDLEY_INTEL_VERSION_CHECK)
    #undef JSON_HEDLEY_INTEL_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_INTEL_VERSION)
    #define JSON_HEDLEY_INTEL_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_INTEL_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_INTEL_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_INTEL_CL_VERSION)
    #undef JSON_HEDLEY_INTEL_CL_VERSION
#endif
#if defined(__INTEL_COMPILER) && defined(__INTEL_COMPILER_UPDATE) && defined(__ICL)
    #define JSON_HEDLEY_INTEL_CL_VERSION JSON_HEDLEY_VERSION_ENCODE(__INTEL_COMPILER, __INTEL_COMPILER_UPDATE, 0)
#endif

#if defined(JSON_HEDLEY_INTEL_CL_VERSION_CHECK)
    #undef JSON_HEDLEY_INTEL_CL_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_INTEL_CL_VERSION)
    #define JSON_HEDLEY_INTEL_CL_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_INTEL_CL_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_INTEL_CL_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_PGI_VERSION)
    #undef JSON_HEDLEY_PGI_VERSION
#endif
#if defined(__PGI) && defined(__PGIC__) && defined(__PGIC_MINOR__) && defined(__PGIC_PATCHLEVEL__)
    #define JSON_HEDLEY_PGI_VERSION JSON_HEDLEY_VERSION_ENCODE(__PGIC__, __PGIC_MINOR__, __PGIC_PATCHLEVEL__)
#endif

#if defined(JSON_HEDLEY_PGI_VERSION_CHECK)
    #undef JSON_HEDLEY_PGI_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_PGI_VERSION)
    #define JSON_HEDLEY_PGI_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_PGI_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_PGI_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_SUNPRO_VERSION)
    #undef JSON_HEDLEY_SUNPRO_VERSION
#endif
#if defined(__SUNPRO_C) && (__SUNPRO_C > 0x1000)
    #define JSON_HEDLEY_SUNPRO_VERSION JSON_HEDLEY_VERSION_ENCODE((((__SUNPRO_C >> 16) & 0xf) * 10) + ((__SUNPRO_C >> 12) & 0xf), (((__SUNPRO_C >> 8) & 0xf) * 10) + ((__SUNPRO_C >> 4) & 0xf), (__SUNPRO_C & 0xf) * 10)
#elif defined(__SUNPRO_C)
    #define JSON_HEDLEY_SUNPRO_VERSION JSON_HEDLEY_VERSION_ENCODE((__SUNPRO_C >> 8) & 0xf, (__SUNPRO_C >> 4) & 0xf, (__SUNPRO_C) & 0xf)
#elif defined(__SUNPRO_CC) && (__SUNPRO_CC > 0x1000)
    #define JSON_HEDLEY_SUNPRO_VERSION JSON_HEDLEY_VERSION_ENCODE((((__SUNPRO_CC >> 16) & 0xf) * 10) + ((__SUNPRO_CC >> 12) & 0xf), (((__SUNPRO_CC >> 8) & 0xf) * 10) + ((__SUNPRO_CC >> 4) & 0xf), (__SUNPRO_CC & 0xf) * 10)
#elif defined(__SUNPRO_CC)
    #define JSON_HEDLEY_SUNPRO_VERSION JSON_HEDLEY_VERSION_ENCODE((__SUNPRO_CC >> 8) & 0xf, (__SUNPRO_CC >> 4) & 0xf, (__SUNPRO_CC) & 0xf)
#endif

#if defined(JSON_HEDLEY_SUNPRO_VERSION_CHECK)
    #undef JSON_HEDLEY_SUNPRO_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_SUNPRO_VERSION)
    #define JSON_HEDLEY_SUNPRO_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_SUNPRO_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_SUNPRO_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_EMSCRIPTEN_VERSION)
    #undef JSON_HEDLEY_EMSCRIPTEN_VERSION
#endif
#if defined(__EMSCRIPTEN__)
    #define JSON_HEDLEY_EMSCRIPTEN_VERSION JSON_HEDLEY_VERSION_ENCODE(__EMSCRIPTEN_major__, __EMSCRIPTEN_minor__, __EMSCRIPTEN_tiny__)
#endif

#if defined(JSON_HEDLEY_EMSCRIPTEN_VERSION_CHECK)
    #undef JSON_HEDLEY_EMSCRIPTEN_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_EMSCRIPTEN_VERSION)
    #define JSON_HEDLEY_EMSCRIPTEN_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_EMSCRIPTEN_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_EMSCRIPTEN_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_ARM_VERSION)
    #undef JSON_HEDLEY_ARM_VERSION
#endif
#if defined(__CC_ARM) && defined(__ARMCOMPILER_VERSION)
    #define JSON_HEDLEY_ARM_VERSION JSON_HEDLEY_VERSION_ENCODE(__ARMCOMPILER_VERSION / 1000000, (__ARMCOMPILER_VERSION % 1000000) / 10000, (__ARMCOMPILER_VERSION % 10000) / 100)
#elif defined(__CC_ARM) && defined(__ARMCC_VERSION)
    #define JSON_HEDLEY_ARM_VERSION JSON_HEDLEY_VERSION_ENCODE(__ARMCC_VERSION / 1000000, (__ARMCC_VERSION % 1000000) / 10000, (__ARMCC_VERSION % 10000) / 100)
#endif

#if defined(JSON_HEDLEY_ARM_VERSION_CHECK)
    #undef JSON_HEDLEY_ARM_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_ARM_VERSION)
    #define JSON_HEDLEY_ARM_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_ARM_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_ARM_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_IBM_VERSION)
    #undef JSON_HEDLEY_IBM_VERSION
#endif
#if defined(__ibmxl__)
    #define JSON_HEDLEY_IBM_VERSION JSON_HEDLEY_VERSION_ENCODE(__ibmxl_version__, __ibmxl_release__, __ibmxl_modification__)
#elif defined(__xlC__) && defined(__xlC_ver__)
    #define JSON_HEDLEY_IBM_VERSION JSON_HEDLEY_VERSION_ENCODE(__xlC__ >> 8, __xlC__ & 0xff, (__xlC_ver__ >> 8) & 0xff)
#elif defined(__xlC__)
    #define JSON_HEDLEY_IBM_VERSION JSON_HEDLEY_VERSION_ENCODE(__xlC__ >> 8, __xlC__ & 0xff, 0)
#endif

#if defined(JSON_HEDLEY_IBM_VERSION_CHECK)
    #undef JSON_HEDLEY_IBM_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_IBM_VERSION)
    #define JSON_HEDLEY_IBM_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_IBM_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_IBM_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_TI_VERSION)
    #undef JSON_HEDLEY_TI_VERSION
#endif
#if \
    defined(__TI_COMPILER_VERSION__) && \
    ( \
      defined(__TMS470__) || defined(__TI_ARM__) || \
      defined(__MSP430__) || \
      defined(__TMS320C2000__) \
    )
#if (__TI_COMPILER_VERSION__ >= 16000000)
    #define JSON_HEDLEY_TI_VERSION JSON_HEDLEY_VERSION_ENCODE(__TI_COMPILER_VERSION__ / 1000000, (__TI_COMPILER_VERSION__ % 1000000) / 1000, (__TI_COMPILER_VERSION__ % 1000))
#endif
#endif

#if defined(JSON_HEDLEY_TI_VERSION_CHECK)
    #undef JSON_HEDLEY_TI_VERSION_CHECK
#endif
#if defined(JSON_HEDLEY_TI_VERSION)
    #define JSON_HEDLEY_TI_VERSION_CHECK(major,minor,patch) (JSON_HEDLEY_TI_VERSION >= JSON_HEDLEY_VERSION_ENCODE(major, minor, patch))
#else
    #define JSON_HEDLEY_TI_VERSION_CHECK(major,minor,patch) (0)
#endif

#if defined(JSON_HEDLEY_TI_CL2000_VERSION)
    #undef JSON_HEDLEY_TI_CL2000_VERSION
#endif
#if defined(__TI_COMPILER_VERSION__) && defined(__TMS320C2000__)
    #define JSON_HEDLEY_TI_CL2000_VERSION JSON_HEDLEY_VERSION_ENCODE(__TI_COMPILER_VERSION__ / 1000000, (__TI_COMPILER_VERSION__ % 1000000) / 1000, (__TI_COMPILER_VERSION__ % 1000))
#endif

#if defined(JSON_HEDLEY_TI_CL2000_VERSION_CHECK)
    #undef JSON_HEDLEY_TI_CL2000_VERSION_CHECK
