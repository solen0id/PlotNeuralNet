import sys

sys.path.append("../")

import pycore.tikzeng as tz


def generate_architecture():

    architecture = [
        tz.make_header(".."),
        tz.make_colors(),
        tz.begin_document(),
        # fmt:off
        tz.inputLayer("input", 1, (16,16), offset="(0,0,0)", origin="(0,0,0)", width=1, height=25, depth=25, caption=r"$x$"),
        tz.strided_conv_lrelu_block("conv1", 8, (8,8), offset="(2,0,0)", origin="(input-east)", width=3, height=20, depth=20, caption=""),
        tz.strided_conv_lrelu_block("conv2", 16, (4,4), offset="(2,0,0)", origin="(conv1_out-east)", width=6, height=15, depth=15, caption=""),
        tz.strided_conv_lrelu_block("conv3", 32, (2,2), offset="(2,0,0)", origin="(conv2_out-east)", width=12, height=10, depth=10, caption=""),
        tz.flatten("flatten", 128, offset="(2,0,0)", origin="(conv3_out-east)", width=10, height=2, depth=2, caption=r"$z$"),
        tz.reshape("reshape", 32, (2,2), offset="(2,0,0)", origin="(flatten-east)", width=12, height=10, depth=10, caption=""),
        tz.deconv2D_strided("deconv1", 16, (4,4), offset="(2,0,0)", origin="(reshape-east)", width=6, height=15, depth=15, caption=""),
        tz.deconv2D_strided("deconv2", 8, (8,8), offset="(2,0,0)", origin="(deconv1-east)", width=3, height=20, depth=20, caption=""),
        tz.strided_deconv_tanh_block("deconv3", 1, (16,16), offset="(2,0,0)", origin="(deconv2-east)", prev_width=3, width=1, height=25, depth=25, caption=r"$\hat{x}$"),
        # fmt:on
        # Connections
        tz.connection("input", "conv1_in"),
        tz.connection("conv1_out", "conv2_in"),
        tz.connection("conv2_out", "conv3_in"),
        tz.connection("conv3_out", "flatten"),
        tz.connection("flatten", "reshape"),
        tz.connection("reshape", "deconv1"),
        tz.connection("deconv1", "deconv2"),
        tz.connection("deconv2", "deconv3_in"),
        tz.create_legend(
            input_=True,
            conv_strided=True,
            deconv_strided=True,
            flatten=True,
            reshape=True,
            leaky_relu=True,
            tanh=True,
        ),
        tz.end_document(),
    ]

    return architecture


if __name__ == "__main__":
    file_name = str(sys.argv[0]).split(".")[0]
    arch = generate_architecture()
    tz.generate_tex_document(arch, f"{file_name}.tex")
