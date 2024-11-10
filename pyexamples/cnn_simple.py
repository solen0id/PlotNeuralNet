import sys

sys.path.append("../")

import pycore.tikzeng as tz


def generate_architecture():

    architecture = [
        tz.make_header(".."),
        tz.make_colors(),
        tz.begin_document(),
        # fmt:off
        tz.inputLayer("input", None, None, offset="(0,0,0)", origin="(0,0,0)", width=1, height=25, depth=25, caption=""),
        tz.conv_lrelu_pool_block("conv1", None, None, offset="(2,0,0)", origin="(input-east)", width=3, height=20, depth=20, caption=""),
        tz.conv_lrelu_pool_block("conv2", None, None, offset="(2,0,0)", origin="(conv1_out-east)", width=6, height=15, depth=15, caption=""),
        tz.conv_lrelu_pool_block("conv3",None, None, offset="(2,0,0)", origin="(conv2_out-east)", width=12, height=10, depth=10, caption=""),
        tz.flatten("flatten", None, offset="(2,0,0)", origin="(conv3_out-east)", width=10, height=2, depth=2, caption=""),
        tz.fully_connected("dense", None, offset="(2,0,0)", origin="(flatten-east)", width=15, height=1, depth=1, caption=""),
        tz.fully_connected("dense2", None, offset="(2,0,0)", origin="(dense-east)", width=5, height=1, depth=1, caption=""),
        tz.fully_connected("dense3", None, offset="(2,0,0)", origin="(dense2-east)", width=2, height=1, depth=1, caption=""),
        tz.softmax("softmax", None, None, offset="(0,0,0)", origin="(dense3-east)", width=2, height=1, depth=1, caption=""),

        tz.connection("input", "conv1_in"),
        tz.connection("conv1_out", "conv2_in"),
        tz.connection("conv2_out", "conv3_in"),
        tz.connection("conv3_out", "flatten"),
        tz.connection("flatten", "dense"),
        tz.connection("dense", "dense2"),
        tz.connection("dense2", "dense3"),




        tz.create_legend(
            input_=True,
            conv=True,
            flatten=True,
            pool=True,
            leaky_relu=True,
            fc=True,
            softmax=True,
        ),
        tz.end_document(),
    ]

    return architecture


if __name__ == "__main__":
    file_name = str(sys.argv[0]).split(".")[0]
    arch = generate_architecture()
    tz.generate_tex_document(arch, f"{file_name}.tex")
