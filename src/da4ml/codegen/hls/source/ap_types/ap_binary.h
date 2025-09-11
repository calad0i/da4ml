/*
 * Copyright 2024-2024 Chang Sun
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __AP_BINARY_H__
#define __AP_BINARY_H__

#include <ap_fixed.h>
#include <cassert>

struct ap_binary {

    bool is_one;

    INLINE ap_binary() {}

    INLINE ap_binary(const bool value) : is_one(value) {}
    INLINE ap_binary(const ap_binary &value) : is_one(value.is_one) {}

    INLINE operator int() const { return is_one ? 1 : -1; }
    INLINE operator float() const { return is_one ? 1.0 : -1.0; }

    template <typename T> INLINE ap_binary(T value) : is_one(value >= 0) {}

    template <typename T>
    INLINE auto operator=(T value) -> decltype(std::enable_if_t<std::is_same<T, ap_binary>::value, int>()) {
        is_one = value.is_one;
        return 0;
    }

    template <typename T>
    INLINE auto operator=(T value) -> decltype(std::enable_if_t<!std::is_same<T, ap_binary>::value, int>()) {
        is_one = value >= 0;
        return 0;
    }

    INLINE ap_fixed<2, 1> value() const { return is_one ? 1 : -1; }

    template <typename T> INLINE bool operator==(T value) const { return value() == value; }

    template <typename T> INLINE bool operator!=(T value) const { return value() != value; }

    template <typename T> INLINE bool operator<(T value) const { return value() < value; }

    template <typename T> INLINE bool operator<=(T value) const { return value() <= value; }

    template <typename T> INLINE bool operator>(T value) const { return value() > value; }

    template <typename T> INLINE bool operator>=(T value) const { return value() >= value; }

    template <typename T> INLINE ap_binary operator+(T value) const { return ap_binary(is_one || value.is_one); }

    template <typename T> INLINE ap_binary operator*(T value) const { return ap_binary(is_one && value.is_one); }

    template <typename T> INLINE ap_binary operator-(T value) const { return ap_binary(is_one && !value.is_one); }

    template <typename T> INLINE T operator+(T value) { return value + value(); }

    template <typename T> INLINE T operator*(T value) { return value * value(); }

    template <typename T> INLINE T operator-(T value) { return value - value(); }
};

typedef ap_fixed<2, 1, AP_RND_CONV, AP_SAT_SYM> ap_ternary;

#endif