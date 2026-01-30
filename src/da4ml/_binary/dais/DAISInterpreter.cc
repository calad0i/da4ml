#include "DAISInterpreter.hh"
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <cmath>

namespace dais {

    void DAISInterpreter::load_from_binary(const std::vector<int32_t> &binary_data) {
        if (binary_data.size() < 6) {
            throw std::runtime_error(
                "Binary data too small to contain valid DAIS model file"
            );
        }
        if (binary_data[0] != dais_version) {
            throw std::runtime_error(
                "DAIS version mismatch: expected version " +
                std::to_string(dais_version) + ", got version " +
                std::to_string(binary_data[0])
            );
        }

        n_in = binary_data[2];
        n_out = binary_data[3];
        n_ops = binary_data[4];
        n_tables = binary_data[5];

        size_t fixed_offset = 6;

        size_t table_offset = fixed_offset + n_in + 3 * n_out + 8 * n_ops;

        size_t expect_length = table_offset;
        if (n_tables > 0) {
            for (size_t i = 0; i < n_tables; ++i) {
                int32_t table_size = binary_data[table_offset + i];
                expect_length += 1 + table_size;
            }
        }

        const static size_t d_size = sizeof(int32_t);

        if (binary_data.size() != expect_length) {
            throw std::runtime_error(
                "Binary data size mismatch: expected " +
                std::to_string(expect_length * d_size) + " bytes , got " +
                std::to_string(binary_data.size() * d_size) + " bytes"
            );
        }

        ops.resize(n_ops);
        inp_shifts.resize(n_in);
        out_idxs.resize(n_out);
        out_shifts.resize(n_out);
        out_negs.resize(n_out);

        std::memcpy(inp_shifts.data(), &binary_data[fixed_offset], n_in * d_size);
        std::memcpy(out_idxs.data(), &binary_data[fixed_offset + n_in], n_out * d_size);
        std::memcpy(
            out_shifts.data(), &binary_data[fixed_offset + n_in + n_out], n_out * d_size
        );
        std::memcpy(
            out_negs.data(), &binary_data[fixed_offset + n_in + 2 * n_out], n_out * d_size
        );
        std::memcpy(
            ops.data(), &binary_data[fixed_offset + n_in + 3 * n_out], n_ops * 8 * d_size
        );

        size_t curr_table_offset = table_offset + n_tables;
        for (size_t i = 0; i < n_tables; ++i) {
            int32_t table_size = binary_data[table_offset + i];
            std::vector<int32_t> table_data(table_size);
            std::memcpy(
                table_data.data(), &binary_data[curr_table_offset], table_size * d_size
            );
            lookup_tables.emplace_back(table_data);
            curr_table_offset += table_size;
        }

        for (const auto &op : ops) {
            int32_t width = op.dtype.width();
            if (op.opcode == -1)
                max_inp_width = std::max(max_inp_width, width);
            max_ops_width = std::max(max_ops_width, width);
        }
        for (const int32_t &idx : out_idxs) {
            if (idx >= 0)
                max_out_width = std::max(max_out_width, ops[idx].dtype.width());
        }
        validate();
    }

    void DAISInterpreter::load_from_file(const std::string &path) {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("Failed to open file: " + path);

        std::vector<int32_t> binary_data;
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        if (file_size % sizeof(int32_t) != 0)
            throw std::runtime_error("File size is not a multiple of int32_t size");
        if (file_size < 3 * sizeof(int32_t))
            throw std::runtime_error(
                "File size is too small to contain valid DAIS model file"
            );
        size_t num_elements = file_size / sizeof(int32_t);
        binary_data.resize(num_elements);
        file.read(reinterpret_cast<char *>(binary_data.data()), file_size);
        load_from_binary(binary_data);
    }

    int64_t DAISInterpreter::shift_add(
        int64_t v1,
        int64_t v2,
        int32_t shift,
        bool is_minus,
        const DType &dtype0,
        const DType &dtype1,
        const DType &dtype_out
    ) const {
        int32_t actual_shift = shift + dtype0.fractionals - dtype1.fractionals;
        int64_t _v2 = is_minus ? -v2 : v2;
        if (actual_shift > 0)
            return v1 + (_v2 << actual_shift);
        else
            return (v1 << -actual_shift) + _v2;
    }

    int64_t DAISInterpreter::quantize(
        int64_t value,
        const DType &dtype_from,
        const DType &dtype_to
    ) const {
        int32_t shift = dtype_from.fractionals - dtype_to.fractionals;
        value = value >> shift;
        int32_t int_max = dtype_to.int_max();
        int32_t int_min = dtype_to.int_min();
        const int64_t _mod = 1LL << dtype_to.width();
        value =
            ((value - int_min + (std::abs(value) / _mod + 1) * _mod) % _mod) + int_min;
        return value;
    }

    int64_t DAISInterpreter::relu(
        int64_t value,
        const DType &dtype_from,
        const DType &dtype_to
    ) const {
        if (value < 0)
            return 0;
        return quantize(value, dtype_from, dtype_to);
    }

    int64_t DAISInterpreter::const_add(
        int64_t value,
        const DType dtype_from,
        const DType dtype_to,
        int32_t data_high,
        int32_t data_low
    ) const {
        const int32_t _shift = dtype_to.fractionals - dtype_from.fractionals;
        int64_t data =
            (static_cast<int64_t>(data_high) << 32) | static_cast<uint32_t>(data_low);
        // std::cout << "v=" << value << " c=" << data << " shift=" << _shift << std::endl;
        return (value << _shift) + data;
    }

    bool DAISInterpreter::get_msb(int64_t value, const DType &dtype) const {
        if (dtype.is_signed)
            return value < 0;
        return value >= (1LL << (dtype.width() - 2));
    }

    int64_t DAISInterpreter::msb_mux(
        int64_t v0,
        int64_t v1,
        int64_t v_cond,
        int32_t _shift,
        const DType &dtype0,
        const DType &dtype1,
        const DType &dtype_cond,
        const DType &dtype_out
    ) const {
        bool cond = get_msb(v_cond, dtype_cond);
        int32_t shift = dtype0.fractionals - dtype1.fractionals + _shift;
        int32_t shift0 = shift > 0 ? 0 : -shift;
        int32_t shift1 = shift > 0 ? shift : 0;
        int64_t result;
        DType dtype_in;

        if (cond) {
            dtype_in = (dtype0 << shift0).with_fractionals(dtype_out.fractionals);
            result = v0 << shift0;
        }
        else {
            dtype_in = (dtype1 << shift1).with_fractionals(dtype_out.fractionals);
            result = v1 << shift1;
        }

        return quantize(result, dtype_in, dtype_out);
    }

    int64_t DAISInterpreter::logic_lookup(
        const int64_t v0,
        const Op &op,
        const DType dtype_in
    ) const {
        int32_t table_idx = op.data_low;
        const auto &table = lookup_tables[table_idx];
        size_t table_size = table.size();
        int64_t zero = -dtype_in.is_signed * (1LL << (dtype_in.width() - 1));

        int64_t index = v0 - zero - op.data_high;
        if (index < 0 || index >= table_size) {
            throw std::runtime_error(
                "Logic lookup index out of bounds: index=" + std::to_string(index) +
                ", table_size=" + std::to_string(table_size) + ", zero=" +
                std::to_string(zero) + ", data_high=" + std::to_string(op.data_high) +
                ", v0=" + std::to_string(v0)
            );
        }
        return static_cast<int64_t>(table[index]);
    }

    std::vector<int64_t>
    DAISInterpreter::exec_ops(const std::span<const double> &inputs) {
        if (inputs.size() != n_in)
            throw std::runtime_error(
                "Input size mismatch: expected " + std::to_string(n_in) + ", got " +
                std::to_string(inputs.size())
            );

        std::vector<int64_t> buffer(n_ops);
        std::vector<int64_t> output_buffer(n_out);

        for (size_t i = 0; i < n_ops; ++i) {
            const Op &op = ops[i];
            switch (op.opcode) {
            case -1: {
                int64_t input_value = static_cast<int64_t>(std::floor(
                    inputs[op.id0] *
                    std::pow(2.0, inp_shifts[op.id0] + ops[i].dtype.fractionals)
                ));
                buffer[i] = quantize(input_value, op.dtype, op.dtype);
                break;
            }
            case 0:
            case 1:
                buffer[i] = shift_add(
                    buffer[op.id0],
                    buffer[op.id1],
                    op.data_low,
                    op.opcode == 1,
                    ops[op.id0].dtype,
                    ops[op.id1].dtype,
                    ops[i].dtype
                );
                break;
            case 2:
            case -2:
                buffer[i] = relu(
                    op.opcode == -2 ? -buffer[op.id0] : buffer[op.id0],
                    ops[op.id0].dtype,
                    ops[i].dtype
                );
                break;
            case 3:
            case -3:
                buffer[i] = quantize(
                    op.opcode == -3 ? -buffer[op.id0] : buffer[op.id0],
                    ops[op.id0].dtype,
                    ops[i].dtype
                );
                break;
            case 4:
                buffer[i] = const_add(
                    buffer[op.id0],
                    ops[op.id0].dtype,
                    ops[i].dtype,
                    op.data_high,
                    op.data_low
                );
                break;
            case 5:
                buffer[i] = static_cast<int64_t>(op.data_high) << 32 |
                            static_cast<uint32_t>(op.data_low);
                break;
            case 6:
            case -6:
                buffer[i] = msb_mux(
                    buffer[op.id0],
                    op.opcode == -6 ? -buffer[op.id1] : buffer[op.id1],
                    buffer[op.data_low],
                    op.data_high,
                    ops[op.id0].dtype,
                    ops[op.id1].dtype,
                    ops[op.data_low].dtype,
                    ops[i].dtype
                );
                break;
            case 7: buffer[i] = buffer[op.id0] * buffer[op.id1]; break;
            case 8:
                buffer[i] = logic_lookup(buffer[op.id0], op, ops[op.id0].dtype);
                break;

            default:
                throw std::runtime_error(
                    "Unknown opcode: " + std::to_string(op.opcode) + " at index " +
                    std::to_string(i)
                );
            }
        }
        for (size_t i = 0; i < n_out; ++i)
            output_buffer[i] = out_idxs[i] >= 0 ? buffer[out_idxs[i]] : 0;
        return output_buffer;
    }

    void DAISInterpreter::inference(
        const std::span<const double> &inputs,
        std::span<double> &outputs
    ) {
        std::vector<int64_t> int_outputs = exec_ops(inputs);
        for (size_t i = 0; i < n_out; ++i) {
            int64_t tmp = out_negs[i] ? -int_outputs[i] : int_outputs[i];
            outputs[i] =
                static_cast<double>(tmp) *
                std::pow(2.0, out_shifts[i] - ops[out_idxs[i]].dtype.fractionals);
        }
    }

    std::vector<double>
    DAISInterpreter::inference(const std::span<const double> &inputs) {
        std::vector<double> outputs(n_out);
        std::span<double> out_span(outputs.data(), n_out);
        inference(inputs, out_span);
        return outputs;
    }

    void DAISInterpreter::print_program_info() const {
        size_t bits_in = 0, bits_out = 0;
        for (int32_t i = 0; i < n_ops; ++i) {
            const Op op = ops[i];
            if (op.opcode == -1)
                bits_in += op.dtype.width();
        }
        for (int32_t i = 0; i < n_out; ++i) {
            if (out_idxs[i] >= 0)
                bits_out += ops[out_idxs[i]].dtype.width();
        }
        std::cout << "DAIS Sequence:\n";
        std::cout << n_in << " (" << bits_in << " bits) -> " << n_out << " (" << bits_out
                  << " bits)\n";
        std::cout << "# operations: " << n_ops << "\n";
        std::cout << "Maximum intermediate width: " << max_ops_width << " bits\n";
    }

    void DAISInterpreter::validate() const {
        for (int32_t i = 0; i < n_ops; ++i) // Causality check
        {
            const Op &op = ops[i];
            if (op.id0 >= i && op.opcode != -1)
                throw std::runtime_error(
                    "Operation " + std::to_string(i) +
                    " has id0=" + std::to_string(op.id0) + "violating causality"
                );
            if (op.id1 >= i)
                throw std::runtime_error(
                    "Operation " + std::to_string(i) +
                    " has id1=" + std::to_string(op.id1) + " violating causality"
                );
            if (abs(op.opcode) == 6 && op.data_low >= i)
                throw std::runtime_error(
                    "Operation " + std::to_string(i) + " has cond_idx=" +
                    std::to_string(op.data_low) + " violating causality"
                );
        }

        if (max_ops_width > 64) {
            std::cerr << "Warning: max_ops_width=" << max_ops_width
                      << " exceeds 64 bits. This may comppromise bit-exactness of the "
                         "interpreter.\n"
                      << "This high wdith is very unusual for a properly quantized "
                         "network, so you may want to check your model.\n";
        }
    }
} // namespace dais
