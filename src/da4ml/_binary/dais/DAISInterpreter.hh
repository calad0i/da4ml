#pragma once

#include <cstdint>
#include <stdalign.h>
#include <string>
#include <vector>
#include <span>

#ifdef _OPENMP
#include <omp.h>
constexpr bool _openmp = true;
#else
constexpr bool _openmp = false;
#endif

namespace dais {

    struct DType {
        int32_t is_signed;
        int32_t integers;
        int32_t fractionals;

        int32_t width() const { return integers + fractionals + (is_signed ? 1 : 0); }
        int32_t int_max() const { return (1 << (width() - (is_signed ? 1 : 0))) - 1; }
        int32_t int_min() const { return is_signed ? -(1 << (width() - 1)) : 0; }

        DType operator<<(int32_t shift) const {
            return DType{is_signed, integers + shift, fractionals - shift};
        }

        DType operator>>(int32_t shift) const {
            return DType{is_signed, integers - shift, fractionals + shift};
        }

        DType with_fractionals(int32_t new_fractionals) const {
            return DType{
                is_signed, integers + (fractionals - new_fractionals), new_fractionals
            };
        }

        DType with_integers(int32_t new_integers) const {
            return DType{
                is_signed, new_integers, fractionals + (integers - new_integers)
            };
        }

        DType with_signed(int32_t new_is_signed) const {
            return DType{new_is_signed, integers, fractionals};
        }
    };

    struct alignas(4) Op {
        int32_t opcode;
        int32_t id0;
        int32_t id1;
        int32_t data_low;
        int32_t data_high;
        DType dtype; // (signed, integer_bits, fractional_bits)
    };

    class DAISInterpreter {
      private:
        int32_t n_in, n_out, n_ops, n_tables;
        int32_t max_ops_width = 0, max_inp_width = 0, max_out_width = 0;
        std::vector<int32_t> inp_shifts;
        std::vector<int32_t> out_idxs;
        std::vector<int32_t> out_shifts;
        std::vector<int32_t> out_negs;
        std::vector<Op> ops;
        std::vector<std::vector<int32_t>> lookup_tables;

        void validate() const;

        // 0, 1
        int64_t shift_add(
            int64_t v1,
            int64_t v2,
            int32_t shift,
            bool sign,
            const DType &dtype0,
            const DType &dtype1,
            const DType &dtype_out
        ) const;

        int64_t const_add(
            int64_t value,
            DType dtype_from,
            DType dtype_to,
            int32_t data_high,
            int32_t data_low
        ) const;

        // 2, -2
        int64_t relu(int64_t value, const DType &dtype_from, const DType &dtype_to) const;

        // 3, -3
        int64_t
        quantize(int64_t value, const DType &dtype_from, const DType &dtype_to) const;

        // 6, -6
        bool get_msb(int64_t value, const DType &dtype) const;

        int64_t msb_mux(
            int64_t v0,
            int64_t v1,
            int64_t v_cond,
            int32_t _shift,
            const DType &dtype0,
            const DType &dtype1,
            const DType &dtype_cond,
            const DType &dtype_out
        ) const;

        std::vector<int64_t> exec_ops(const std::span<const double> &inputs);

        // 8
        int64_t logic_lookup(int64_t v1, const Op &op, const DType dtype_in) const;

        int64_t bit_unary(int64_t v, const Op &op) const;

        int64_t bit_binary(int64_t v1, int64_t v2, const Op &op) const;

      public:
        static const int dais_version = 0;

        void load_from_file(const std::string &filename);

        void load_from_binary(const std::span<const int32_t> &binary_data);

        std::vector<double> inference(const std::span<const double> &inputs);
        void inference(const std::span<const double> &inputs, std::span<double> &outputs);

        void print_program_info() const;
    };

} // namespace dais
