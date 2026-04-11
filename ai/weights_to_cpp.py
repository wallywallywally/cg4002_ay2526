import os
from pathlib import Path

import torch

script_dir = Path(__file__).resolve().parent
MODEL_VERSION = "2.3"
DEFAULT_CNN_INPUT = script_dir / f"v{MODEL_VERSION}" / "cnn_weights.pth"
DEFAULT_CNN_OUTPUT = script_dir / f"v{MODEL_VERSION}" / "weights.h"


def _format_value(value):
    return f"{float(value):.8f}"


def _format_1d_array(values):
    return "{" + ", ".join(_format_value(value) for value in values) + "}"


def _format_2d_array(values):
    return "{" + ", ".join(_format_1d_array(row) for row in values) + "}"


def _write_header_prelude(handle, guard_name, include_name):
    handle.write(f"#ifndef {guard_name}\n")
    handle.write(f"#define {guard_name}\n\n")
    handle.write(f'#include "{include_name}"\n\n')


def export_cnn_weights_from_path(input_path=DEFAULT_CNN_INPUT, output_path=DEFAULT_CNN_OUTPUT):
    model_data = torch.load(input_path, map_location="cpu")

    def to_cpp(tensor, name):
        flattened = tensor.detach().cpu().numpy().flatten()
        return f"const data_t {name}[] = " + _format_1d_array(flattened) + ";"

    print("Checking CNN weights:")
    for key in model_data.keys():
        print(f"{key}: {len(model_data[key].detach().cpu().numpy().flatten())}")

    with open(output_path, "w", encoding="utf-8") as handle:
        _write_header_prelude(handle, "WEIGHT_H", "cnn_top.h")
        for key in model_data.keys():
            handle.write(to_cpp(model_data[key], key.replace(".", "_")) + "\n")
        handle.write("#endif\n")

    print(f"Weights saved to {output_path}")


def _fold_batch_norm(linear_weight, linear_bias, batch_norm_module):
    eps = batch_norm_module.eps
    gamma = batch_norm_module.weight.detach().cpu()
    beta = batch_norm_module.bias.detach().cpu()
    mean = batch_norm_module.running_mean.detach().cpu()
    var = batch_norm_module.running_var.detach().cpu()

    scale = gamma / torch.sqrt(var + eps)
    folded_weight = linear_weight.detach().cpu() * scale[:, None]
    folded_bias = beta + (linear_bias.detach().cpu() - mean) * scale
    return folded_weight, folded_bias


def export_mlp_weights(model, output_path):
    model.eval()

    if not hasattr(model, "network"):
        raise ValueError("Expected an MLPClassifier instance with a 'network' attribute.")

    network = model.network
    if len(network) < 10:
        raise ValueError("Unexpected MLP layout. Expected Flatten -> LazyLinear -> BatchNorm1d -> Linear -> Linear -> Linear.")

    first_linear = network[1]
    batch_norm = network[2]
    second_linear = network[5]
    third_linear = network[7]
    fourth_linear = network[9]

    if not hasattr(first_linear, "weight"):
        raise ValueError("MLP first linear layer is not initialized yet. Run a forward pass before exporting weights.")

    fc1_weight, fc1_bias = _fold_batch_norm(first_linear.weight, first_linear.bias, batch_norm)

    fc2_weight = second_linear.weight.detach().cpu()
    fc2_bias = second_linear.bias.detach().cpu()
    fc3_weight = third_linear.weight.detach().cpu()
    fc3_bias = third_linear.bias.detach().cpu()
    fc4_weight = fourth_linear.weight.detach().cpu()
    fc4_bias = fourth_linear.bias.detach().cpu()

    with open(output_path, "w", encoding="utf-8") as handle:
        _write_header_prelude(handle, "MLP_WEIGHTS_H", "mlp_top.h")
        handle.write(
            f"static const mlp_data_t mlp_fc1_weight[MLP_H1_SIZE][MLP_INPUT_SIZE] = "
            f"{_format_2d_array(fc1_weight.numpy())};\n"
        )
        handle.write(
            f"static const mlp_data_t mlp_fc1_bias[MLP_H1_SIZE] = "
            f"{_format_1d_array(fc1_bias.numpy())};\n"
        )
        handle.write(
            f"static const mlp_data_t mlp_fc2_weight[MLP_H2_SIZE][MLP_H1_SIZE] = "
            f"{_format_2d_array(fc2_weight.numpy())};\n"
        )
        handle.write(
            f"static const mlp_data_t mlp_fc2_bias[MLP_H2_SIZE] = "
            f"{_format_1d_array(fc2_bias.numpy())};\n"
        )
        handle.write(
            f"static const mlp_data_t mlp_fc3_weight[MLP_H3_SIZE][MLP_H2_SIZE] = "
            f"{_format_2d_array(fc3_weight.numpy())};\n"
        )
        handle.write(
            f"static const mlp_data_t mlp_fc3_bias[MLP_H3_SIZE] = "
            f"{_format_1d_array(fc3_bias.numpy())};\n"
        )
        handle.write(
            f"static const mlp_data_t mlp_fc4_weight[MLP_NUM_CLASSES][MLP_H3_SIZE] = "
            f"{_format_2d_array(fc4_weight.numpy())};\n"
        )
        handle.write(
            f"static const mlp_data_t mlp_fc4_bias[MLP_NUM_CLASSES] = "
            f"{_format_1d_array(fc4_bias.numpy())};\n"
        )
        handle.write("\n#endif\n")

    print(f"MLP weights saved to {output_path}")


def export_lstm_weights(model, output_path):
    model.eval()

    if not hasattr(model, "lstm") or not hasattr(model, "fc"):
        raise ValueError("Expected an LSTMClassifier instance with 'lstm' and 'fc' attributes.")

    if getattr(model, "num_layers", None) != 2:
        raise ValueError("The current HLS LSTM exporter expects exactly 2 layers.")

    state = model.state_dict()

    with open(output_path, "w", encoding="utf-8") as handle:
        _write_header_prelude(handle, "LSTM_WEIGHTS_H", "lstm_top.h")

        for layer_idx in range(2):
            w_ih = state[f"lstm.weight_ih_l{layer_idx}"].detach().cpu()
            w_hh = state[f"lstm.weight_hh_l{layer_idx}"].detach().cpu()
            b_ih = state[f"lstm.bias_ih_l{layer_idx}"].detach().cpu()
            b_hh = state[f"lstm.bias_hh_l{layer_idx}"].detach().cpu()

            input_size = w_ih.shape[1]
            handle.write(
                f"static const lstm_data_t lstm_weight_ih_l{layer_idx}[LSTM_GATE_SIZE][{input_size}] = "
                f"{_format_2d_array(w_ih.numpy())};\n"
            )
            handle.write(
                f"static const lstm_data_t lstm_weight_hh_l{layer_idx}[LSTM_GATE_SIZE][LSTM_HIDDEN_SIZE] = "
                f"{_format_2d_array(w_hh.numpy())};\n"
            )
            handle.write(
                f"static const lstm_data_t lstm_bias_ih_l{layer_idx}[LSTM_GATE_SIZE] = "
                f"{_format_1d_array(b_ih.numpy())};\n"
            )
            handle.write(
                f"static const lstm_data_t lstm_bias_hh_l{layer_idx}[LSTM_GATE_SIZE] = "
                f"{_format_1d_array(b_hh.numpy())};\n"
            )

        fc_weight = state["fc.weight"].detach().cpu()
        fc_bias = state["fc.bias"].detach().cpu()
        handle.write(
            f"static const lstm_data_t lstm_fc_weight[LSTM_NUM_CLASSES][LSTM_HIDDEN_SIZE] = "
            f"{_format_2d_array(fc_weight.numpy())};\n"
        )
        handle.write(
            f"static const lstm_data_t lstm_fc_bias[LSTM_NUM_CLASSES] = "
            f"{_format_1d_array(fc_bias.numpy())};\n"
        )

        handle.write("\n#endif\n")

    print(f"LSTM weights saved to {output_path}")


def export_hls_weights(model, output_dir, model_kind=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kind = (model_kind or model.__class__.__name__).lower()
    if "mlp" in kind:
        export_mlp_weights(model, output_dir / "mlp_weights.h")
        return output_dir / "mlp_weights.h"

    if "lstm" in kind:
        export_lstm_weights(model, output_dir / "lstm_weights.h")
        return output_dir / "lstm_weights.h"

    if "cnn" in kind:
        export_cnn_weights_from_path()
        return DEFAULT_CNN_OUTPUT

    raise ValueError(f"Unsupported model kind: {model.__class__.__name__}")


if __name__ == "__main__":
    export_cnn_weights_from_path()