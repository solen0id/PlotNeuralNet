import sys

sys.path.append("../")

import pycore.tikzeng as tz


def generate_architecture():

    architecture = [
        tz.make_header(".."),
        tz.make_colors(),
        tz.begin_document(),
        # Encoder
        tz.inputLayer(
            "input",
            None,
            None,
            offset="(0,0,0)",
            origin="(0,0,0)",
            width=1,
            height=25,
            depth=25,
            caption=r"$x$",
        ),
        tz.conv2D_strided(
            "conv1",
            None,
            None,
            offset="(2,0,0)",
            origin="(input-east)",
            width=3,
            height=20,
            depth=20,
            caption="",
        ),
        tz.conv2D_strided(
            "conv2",
            None,
            None,
            offset="(2,0,0)",
            origin="(conv1-east)",
            width=6,
            height=15,
            depth=15,
            caption="",
        ),
        tz.flatten(
            "flatten",
            None,
            offset="(2,0,0)",
            origin="(conv2-east)",
            width=10,
            height=2,
            depth=2,
            caption="",
        ),
        # Latent space
        tz.fully_connected(
            "mean",
            None,
            offset="(1,1.5,0)",
            origin="(flatten-east)",
            width=10,
            height=2,
            depth=2,
            caption=r"$\mu$",
        ),
        tz.fully_connected(
            "logvar",
            None,
            offset="(1,-1.5,0)",
            origin="(flatten-east)",
            width=10,
            height=2,
            depth=2,
            caption=r"$\log(\sigma^2)$",
        ),
        tz.fully_connected(
            "z",
            None,
            offset="(1,-1.5,0)",
            origin="(mean-east)",
            width=10,
            height=2,
            depth=2,
            caption=r"$z$",
        ),
        # Decoder
        tz.reshape(
            "deconv1",
            None,
            None,
            offset="(2,0,0)",
            origin="(z-east)",
            width=6,
            height=15,
            depth=15,
            caption="",
        ),
        tz.deconv2D_strided(
            "deconv2",
            None,
            None,
            offset="(2,0,0)",
            origin="(deconv1-east)",
            width=3,
            height=20,
            depth=20,
            caption="",
        ),
        tz.deconv2D_strided(
            "output",
            None,
            None,
            offset="(2,0,0)",
            origin="(deconv2-east)",
            width=1,
            height=25,
            depth=25,
            caption=r"$\hat{x}$",
        ),
        # Connections
        tz.connection("input", "conv1"),
        tz.connection("conv1", "conv2"),
        tz.connection("conv2", "flatten"),
        tz.connection("flatten", "mean"),
        tz.connection("flatten", "logvar"),
        tz.connection("mean", "z"),
        tz.connection("logvar", "z"),
        tz.connection("z", "deconv1"),
        tz.connection("deconv1", "deconv2"),
        tz.connection("deconv2", "output"),
        tz.create_legend(
            input_=True,
            conv_strided=True,
            flatten=True,
            fc=True,
            deconv_strided=True,
            reshape=True,
        ),
        tz.end_document(),
    ]

    return architecture


if __name__ == "__main__":
    file_name = str(sys.argv[0]).split(".")[0]
    arch = generate_architecture()
    tz.generate_tex_document(arch, f"{file_name}.tex")
